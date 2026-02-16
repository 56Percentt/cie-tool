from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast
from urllib.parse import urlparse

from rapidfuzz.fuzz import token_set_ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


@dataclass
class MatchConfig:
    w_name_fuzzy: float
    w_name_tfidf: float
    w_domain: float
    thr_high: float
    thr_medium: float
    thr_low: float


def _domain_match(input_domain: Optional[str], candidate_website: Optional[str]) -> float:
    """Returns 1.0 if input domain matches/contained in candidate website; else 0.0."""
    if not input_domain or not candidate_website:
        return 0.0
    inp = input_domain.lower().strip()
    cand = candidate_website.lower().strip()
    if not inp or not cand:
        return 0.0
    return 1.0 if (inp == cand or inp in cand) else 0.0


def _probability_like(score_0_1: float) -> float:
    # MVP monotonic mapping; later replace with calibration (Platt / isotonic).
    s = max(0.0, min(1.0, score_0_1))
    return s


def _token_count(s: str) -> int:
    return len([t for t in s.split() if t])


def _length_penalized_token_set_ratio(a: str, b: str) -> Dict[str, float]:
    """Token-set similarity with a soft penalty for candidates that add extra tokens.

    Problem: token_set_ratio("intel", "intel c++ compiler") == 100 (subset match)
    which can cause products/sub-entities to tie with the parent company.

    We keep the robustness of token_set_ratio, but multiply by:
      sqrt(len(tokens(a)) / len(tokens(b)))  (capped at 1.0)

    This preserves high scores for exact/near-exact names while preferring
    shorter, more "name-like" candidates when the input is short.
    """
    if not a or not b:
        return {"raw": 0.0, "penalty": 0.0, "value": 0.0}

    raw = token_set_ratio(a, b) / 100.0
    ta = _token_count(a)
    tb = _token_count(b)
    if ta == 0 or tb == 0:
        return {"raw": raw, "penalty": 0.0, "value": 0.0}

    ratio = ta / tb
    penalty = 1.0 if ratio >= 1.0 else math.sqrt(ratio)
    return {"raw": raw, "penalty": penalty, "value": raw * penalty}


def _parse_host_and_path(candidate_website: str) -> Dict[str, Any]:
    """Parse a candidate website string into host + path segments.

    Accepts either full URLs (https://example.com/a/b) or bare domains (example.com).
    """
    s = (candidate_website or "").strip()
    if not s:
        return {"host": "", "segments": 0, "url_len": 0, "subdomains": 0, "is_home": False}

    # urlparse treats bare domains as path; add a scheme to normalize.
    parsed = urlparse(s if "://" in s else f"http://{s}")
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]

    segs = [p for p in (parsed.path or "").split("/") if p]
    segments = len(segs)
    labels = [p for p in host.split(".") if p]
    subdomains = max(0, len(labels) - 2) if host else 0
    is_home = segments == 0 and (parsed.path in ("", "/"))

    return {
        "host": host,
        "segments": segments,
        "url_len": len(s),
        "subdomains": subdomains,
        "is_home": is_home,
    }


def _website_tiebreak(normalized_domain: Optional[str], candidate_website: Optional[str]) -> float:
    """Secondary ordering signal to break score ties.

    Prefer "homepage-like" URLs:
      - exact host match to the input domain
      - fewer path segments
      - fewer subdomains
      - shorter URLs
    """
    if not candidate_website:
        return 0.0

    info = _parse_host_and_path(candidate_website)
    host = info["host"]
    segments = info["segments"]
    subdomains = info["subdomains"]
    url_len = info["url_len"]

    nd = (normalized_domain or "").lower().strip()

    exact_host = 1.0 if (nd and host == nd) else 0.0

    # higher is better
    t = 0.0
    t += 2.0 * exact_host
    t += 1.0 / (1.0 + segments)
    t += 1.0 / (1.0 + subdomains)
    t += 1.0 / (1.0 + (url_len / 50.0))
    return t


def score_candidates(
    normalized_name: str,
    normalized_domain: Optional[str],
    candidates: List[Dict[str, Any]],
    cfg: MatchConfig
) -> List[Dict[str, Any]]:
    cand_names = [(c.get("label") or "") for c in candidates]
    cand_names_norm = [(" ".join(str(x).lower().split())).strip() for x in cand_names]

    texts = [normalized_name] + cand_names_norm
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = cast(csr_matrix, vectorizer.fit_transform(texts))
    sims = cosine_similarity(X.getrow(0), X[1:]).ravel() if len(candidates) else []

    scored: List[Dict[str, Any]] = []
    for i, c in enumerate(candidates):
        fuzzy_parts = (
            _length_penalized_token_set_ratio(normalized_name, cand_names_norm[i])
            if normalized_name else
            {"raw": 0.0, "penalty": 0.0, "value": 0.0}
        )
        fuzzy = float(fuzzy_parts["value"])
        tfidf = float(sims[i]) if len(candidates) else 0.0
        dom = _domain_match(normalized_domain, c.get("candidate_website"))

        score = (
            cfg.w_name_fuzzy * fuzzy +
            cfg.w_name_tfidf * tfidf +
            cfg.w_domain * dom
        )
        score = max(0.0, min(1.0, score))

        tiebreak = _website_tiebreak(normalized_domain, c.get("candidate_website"))

        scored.append({
            **c,
            "features": {
                "name_fuzzy": fuzzy,
                "name_fuzzy_raw": float(fuzzy_parts["raw"]),
                "name_len_penalty": float(fuzzy_parts["penalty"]),
                "name_tfidf": tfidf,
                "domain_match": dom,
            },
            "score": score,
            "confidence": _probability_like(score),
            "_tiebreak": tiebreak,
        })

    # Primary sort by score; secondary by homepage-like website characteristics.
    scored.sort(key=lambda x: (x["score"], x.get("_tiebreak", 0.0)), reverse=True)

    # Internal-only field; keep it out of any CSV write by downstream code.
    for s in scored:
        s.pop("_tiebreak", None)

    return scored


def classify_status(conf: float, cfg: MatchConfig) -> str:
    if conf >= cfg.thr_high:
        return "HIGH"
    if conf >= cfg.thr_medium:
        return "MEDIUM"
    if conf >= cfg.thr_low:
        return "LOW"
    # IMPORTANT: "NO_MATCH" is reserved for cases where we truly found nothing viable.
    # If we have a candidate but its confidence is below the LOW threshold, we still
    # consider it a match (weak) so we can surface enrichment with appropriate caution.
    return "WEAK"


def match_reason(top_features: Dict[str, float]) -> str:
    if top_features.get("domain_match", 0.0) >= 1.0:
        return "domain_match"
    if top_features.get("name_fuzzy", 0.0) >= 0.90:
        return "strong_name_match"
    return "ambiguous_name"