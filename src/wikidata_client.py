from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .utils import RateLimiter, SimpleJsonCache


class WikidataClient:
    """
    Connector:
    - Candidate retrieval:
        - Name search via wbsearchentities
        - Domain search via SPARQL (P856 contains domain)
    - Enrichment:
        - Full enrichment for top match via SPARQL
        - Minimal enrichment for candidate scoring (website only)

    Notes:
    - Stock exchange is Wikidata P414 (item -> label)
    - Ticker is usually stored as P249 *qualifier* on the P414 statement, so we fetch it via p:/ps:/pq:
        ?item p:P414 ?stmt .
        ?stmt ps:P414 ?stockExchange .
        ?stmt pq:P249 ?tickerSymbol .
    """

    def __init__(
        self,
        user_agent: str,
        throttle_seconds: float,
        cache: SimpleJsonCache,
        connect_timeout_s: float = 10.0,
        read_timeout_s: float = 120.0,
        retries_total: int = 5,
        backoff_factor: float = 1.5,
    ):
        self.session = requests.Session()

        ua = (user_agent or "").strip() or "cie_tool/1.0 (requests; contact: you@example.com)"
        self.session.headers.update({"User-Agent": ua})

        retry = Retry(
            total=retries_total,
            connect=retries_total,
            read=retries_total,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        self.limiter = RateLimiter(min_interval_s=float(throttle_seconds))
        self.cache = cache

        self._timeout: Tuple[float, float] = (float(connect_timeout_s), float(read_timeout_s))

    def _get_json(
        self,
        url: str,
        params: Dict[str, Any],
        timeout: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        self.limiter.wait()
        try:
            r = self.session.get(url, params=params, timeout=timeout or self._timeout)
            if r.status_code >= 400:
                return None
            return r.json()
        except requests.RequestException:
            return None
        except ValueError:
            return None

    # ---------------------------
    # Candidate search: name
    # ---------------------------
    def search_candidates(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []

        cache_key = f"wd_search::{q}::{top_k}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "language": "en",
            "format": "json",
            "search": q,
            "limit": int(top_k),
        }

        data = self._get_json(url, params=params, timeout=(10.0, 30.0))
        if data is None:
            return []

        results = []
        for item in data.get("search", []) or []:
            results.append(
                {
                    "entity_id": item.get("id"),
                    "label": item.get("label"),
                    "description": item.get("description"),
                    "url": item.get("concepturi"),
                    "source": "wikidata_search",
                    "candidate_website": None,
                }
            )

        self.cache.set(cache_key, results)
        return results

    # ---------------------------
    # Candidate search: domain (SPARQL)
    # ---------------------------
    def search_candidates_by_domain(self, domain: str, top_k: int = 5) -> List[Dict[str, Any]]:
        d = (domain or "").strip().lower()
        if not d:
            return []

        cache_key = f"wd_domain::{d}::{top_k}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        query = f"""
        SELECT ?item ?itemLabel ?desc ?website ?sitelinks WHERE {{
          ?item wdt:P856 ?website .
          FILTER(CONTAINS(LCASE(STR(?website)), "{d}"))

          FILTER EXISTS {{
            ?item wdt:P31/wdt:P279* ?cls .
            VALUES ?cls {{ wd:Q43229 wd:Q783794 wd:Q4830453 }}
          }}

          OPTIONAL {{
            ?item schema:description ?desc .
            FILTER(lang(?desc) = "en")
          }}
          OPTIONAL {{ ?item wikibase:sitelinks ?sitelinks . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY DESC(?sitelinks)
        LIMIT {int(top_k)}
        """

        url = "https://query.wikidata.org/sparql"
        data = self._get_json(url, params={"format": "json", "query": query}, timeout=self._timeout)
        if data is None:
            return []

        results = []
        for b in (data.get("results", {}).get("bindings") or []):
            qid = (b.get("item", {}).get("value", "") or "").split("/")[-1]
            if not qid:
                continue
            results.append(
                {
                    "entity_id": qid,
                    "label": b.get("itemLabel", {}).get("value"),
                    "description": b.get("desc", {}).get("value"),
                    "candidate_website": b.get("website", {}).get("value"),
                    "url": f"https://www.wikidata.org/wiki/{qid}",
                    "source": "wikidata_domain",
                }
            )

        self.cache.set(cache_key, results)
        return results

    # ---------------------------
    # Type guardrail: organization/company check
    # ---------------------------
    def orgish_flags(self, qids: List[str]) -> Optional[Dict[str, bool]]:
        ids = [q.strip() for q in (qids or []) if q and str(q).strip()]
        if not ids:
            return {}

        out: Dict[str, bool] = {}
        missing: List[str] = []

        for qid in ids:
            ck = f"wd_orgish_v1::{qid}"
            cached = self.cache.get(ck)
            if cached is None:
                missing.append(qid)
            else:
                out[qid] = bool(cached)

        if missing:
            values = " ".join(f"wd:{qid}" for qid in missing)
            query = f"""
            SELECT ?item (COUNT(?cls) AS ?cnt) WHERE {{
              VALUES ?item {{ {values} }}
              OPTIONAL {{
                ?item wdt:P31/wdt:P279* ?cls .
                VALUES ?cls {{ wd:Q43229 wd:Q783794 wd:Q4830453 }}
              }}
            }}
            GROUP BY ?item
            """
            url = "https://query.wikidata.org/sparql"
            data = self._get_json(url, params={"format": "json", "query": query}, timeout=(10.0, 45.0))
            if data is None:
                return None

            bindings = (((data.get("results") or {}).get("bindings")) or [])
            seen_qids = set()
            for b in bindings:
                item_uri = ((b.get("item") or {}).get("value") or "")
                qid = item_uri.rsplit("/", 1)[-1] if item_uri else ""
                if not qid:
                    continue
                cnt = ((b.get("cnt") or {}).get("value") or "0")
                try:
                    is_org = int(cnt) > 0
                except Exception:
                    is_org = False
                out[qid] = is_org
                seen_qids.add(qid)
                self.cache.set(f"wd_orgish_v1::{qid}", is_org)

            for qid in missing:
                if qid not in seen_qids:
                    out[qid] = False
                    self.cache.set(f"wd_orgish_v1::{qid}", False)

        return out

    # ---------------------------
    # Minimal enrichment (website only)
    # ---------------------------
    def enrich_entity_minimal(self, qid: str) -> Dict[str, Any]:
        qid = (qid or "").strip()
        if not qid:
            return {}

        cache_key = f"wd_min::{qid}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        query = f"""
        SELECT ?website WHERE {{
          OPTIONAL {{ wd:{qid} wdt:P856 ?website . }}
        }}
        LIMIT 1
        """

        url = "https://query.wikidata.org/sparql"
        data = self._get_json(url, params={"format": "json", "query": query}, timeout=self._timeout)
        if data is None:
            return {}

        website = None
        bindings = (data.get("results", {}).get("bindings") or [])
        row = bindings[0] if bindings else {}
        website = row.get("website", {}).get("value") if row.get("website") else None

        out = {"matched_website": website}
        self.cache.set(cache_key, out)
        return out

    # ---------------------------
    # Full enrichment
    # ---------------------------
    def enrich_entity(self, qid: str) -> Dict[str, Any]:
        qid = (qid or "").strip()
        if not qid:
            return {}

        # v5: output schema drops ticker_symbol and keeps stock_ticker only.
        cache_key = f"wd_enrich_v5::{qid}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        query = f"""
        SELECT ?officialName ?website ?countryLabel ?continentLabel ?hqLabel ?industryLabel
               ?founded ?employees ?parentLabel ?tickerSymbol ?stockExchangeLabel ?desc ?instanceLabel WHERE {{

          BIND(wd:{qid} AS ?item)

          OPTIONAL {{
            ?item wdt:P1448 ?officialName .
            FILTER(LANGMATCHES(LANG(?officialName), "en"))
          }}
          OPTIONAL {{ ?item wdt:P856 ?website . }}

          OPTIONAL {{
            ?item wdt:P17 ?country .
            OPTIONAL {{ ?country wdt:P30 ?continent . }}
          }}

          OPTIONAL {{ ?item wdt:P159 ?hq . }}
          OPTIONAL {{ ?item wdt:P452 ?industry . }}
          OPTIONAL {{ ?item wdt:P571 ?founded . }}
          OPTIONAL {{ ?item wdt:P1128 ?employees . }}
          OPTIONAL {{ ?item wdt:P749 ?parent . }}
          OPTIONAL {{ ?item wdt:P31 ?instance . }}

          # Stock listing: ticker symbol (P249) is usually a QUALIFIER on the stock exchange (P414) statement.
          OPTIONAL {{
            {{
              SELECT ?stockExchange ?tickerSymbol WHERE {{
                VALUES ?item {{ wd:{qid} }}
                ?item p:P414 ?stmt .
                ?stmt ps:P414 ?stockExchange .
                OPTIONAL {{ ?stmt pq:P249 ?tickerSymbol . }}
                OPTIONAL {{ ?stmt pq:P582 ?listingEnd . }}
                OPTIONAL {{ ?stmt pq:P580 ?listingStart . }}
              }}
              # Prefer a "current" listing: no end time first; then most recent start time
              ORDER BY ASC(BOUND(?listingEnd)) DESC(?listingStart)
              LIMIT 1
            }}
          }}

          OPTIONAL {{
            ?item schema:description ?desc .
            FILTER (lang(?desc) = "en")
          }}

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 1
        """

        url = "https://query.wikidata.org/sparql"
        data = self._get_json(url, params={"format": "json", "query": query}, timeout=self._timeout)
        if data is None:
            return {}

        bindings = (data.get("results", {}).get("bindings") or [])
        row = bindings[0] if bindings else {}

        def v(key: str) -> Optional[str]:
            obj = row.get(key)
            return obj.get("value") if obj else None

        out = {
            "matched_entity_id": qid,
            "matched_official_name": v("officialName"),
            "matched_website": v("website"),
            "country": v("countryLabel"),
            "continent": v("continentLabel"),
            "city_hq": v("hqLabel"),
            "industry_raw": v("industryLabel"),
            "description": v("desc"),
            "instance_of": v("instanceLabel"),
            "founded_year": (v("founded") or "")[:4] if v("founded") else None,
            "employee_count": v("employees"),
            "parent_company": v("parentLabel"),

            # Keep only these output-facing listing fields:
            "stock_exchange": v("stockExchangeLabel"),
            "stock_ticker": v("tickerSymbol"),

            "provenance": {
                "matched_official_name": ["wikidata"],
                "matched_website": ["wikidata"],
                "country": ["wikidata"],
                "continent": ["wikidata"],
                "city_hq": ["wikidata"],
                "industry_raw": ["wikidata"],
                "description": ["wikidata"],
                "instance_of": ["wikidata"],
                "founded_year": ["wikidata"],
                "employee_count": ["wikidata"],
                "parent_company": ["wikidata"],
                "stock_exchange": ["wikidata"],
                "stock_ticker": ["wikidata"],
            },
        }

        self.cache.set(cache_key, out)
        return out