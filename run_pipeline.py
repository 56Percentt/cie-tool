from __future__ import annotations

import json
import os
import time

import pandas as pd
import yaml

from src.preprocess import preprocess_input_csv
from src.utils import SimpleJsonCache
from src.wikidata_client import WikidataClient
from src.matching import MatchConfig, score_candidates, classify_status, match_reason
from src.enrichment import build_outputs


_GEO_TOKENS = {
    "afghanistan","albania","algeria","angola","argentina","armenia","australia","austria","azerbaijan",
    "bahrain","bangladesh","belarus","belgium","bolivia","brazil","bulgaria","cambodia","cameroon","canada",
    "chile","china","colombia","croatia","cuba","cyprus","czech","denmark","dominican","ecuador","egypt",
    "ethiopia","finland","france","gabon","georgia","germany","ghana","greece","guatemala","haiti","honduras",
    "hong","kong","hungary","iceland","india","indonesia","iran","iraq","ireland","israel","italy","jamaica",
    "japan","jordan","kazakhstan","kenya","kuwait","kyrgyzstan","laos","latvia","lebanon","libya","lithuania",
    "luxembourg","macedonia","madagascar","malaysia","maldives","mali","malta","mexico","moldova","mongolia",
    "morocco","mozambique","myanmar","namibia","nepal","netherlands","new","zealand","nicaragua","niger","nigeria",
    "norway","oman","pakistan","panama","paraguay","peru","philippines","poland","portugal","qatar","romania",
    "russia","rwanda","saudi","arabia","senegal","serbia","singapore","slovakia","slovenia","somalia","south",
    "africa","spain","sri","lanka","sudan","sweden","switzerland","syria","taiwan","tajikistan","tanzania",
    "thailand","tunisia","turkey","uganda","ukraine","united","arab","emirates","uae","united","kingdom","uk",
    "united","states","usa","uruguay","uzbekistan","venezuela","vietnam","yemen","zambia","zimbabwe",
}

def _fallback_name_queries(normalized_name: str, original_name: str) -> list[str]:
    q = (normalized_name or "").strip()
    orig = (original_name or "").strip()
    out: list[str] = []
    if orig and orig.lower() != q.lower():
        out.append(orig)

    tokens = [t for t in q.split() if t]
    if len(tokens) >= 3:
        out.append(" ".join(tokens[:-1]))

    if tokens:
        trimmed = tokens[:]
        while trimmed and trimmed[-1] in _GEO_TOKENS:
            trimmed = trimmed[:-1]
        no_geo = " ".join(trimmed).strip()
        if no_geo and no_geo.lower() != q.lower():
            out.append(no_geo)
        if no_geo and not no_geo.lower().startswith("the "):
            out.append("the " + no_geo)

    if q and not q.lower().startswith("the "):
        out.append("the " + q)

    seen = set()
    uniq: list[str] = []
    for s in out:
        s2 = (s or "").strip()
        if not s2:
            continue
        key = s2.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s2)
        if len(uniq) >= 4:
            break
    return uniq


_ORG_POS_HINTS = (
    "company", "corporation", "multinational", "public company", "private company",
    "bank", "airline", "telecommunications", "telecom", "insurance", "financial",
    "retailer", "manufacturer", "manufacturing", "holding company", "enterprise",
    "organization", "organisation", "firm", "startup", "subsidiary",
)

_NON_ORG_NEG_HINTS = (
    "fruit", "unit", "si unit", "metric unit", "set of letters", "alphabet",
    "letter", "chemical element", "genus", "species", "food", "plant",
)

def _looks_like_org_by_text(label: str, description: str) -> bool:
    l = (label or "").lower()
    d = (description or "").lower()

    if any(k in d for k in _NON_ORG_NEG_HINTS) and not any(k in d for k in _ORG_POS_HINTS):
        return False

    if any(k in d for k in _ORG_POS_HINTS):
        return True

    if any(t in l.split() for t in ("inc", "inc.", "corp", "corp.", "ltd", "ltd.", "plc", "group", "bank", "airways")):
        return True

    return False


DATA_DICTIONARY = [
    ("input_company_name", "Original company name from input CSV.", "string"),
    ("input_website", "Original website from input CSV (may be blank).", "string/blank"),
    ("normalized_name", "Normalized company name used for retrieval/matching.", "string"),
    ("normalized_domain", "Domain extracted from input website (if provided).", "string/blank"),

    ("match_status", "Confidence bucket: HIGH/MEDIUM/LOW/WEAK. NO_MATCH means no viable candidate was found.", "categorical"),
    ("match_confidence", "Probability-like match score (0–1).", "float"),
    ("match_reason_top", "Reason label for top match (domain_match/strong_name_match/ambiguous_name).", "string"),
    ("matched_entity_id", "Wikidata entity id (QID) for selected match.", "string/blank"),
    ("matched_official_name", "Best available official name for matched entity.", "string/blank"),
    ("matched_website", "Matched entity website (best-effort).", "string/blank"),
    ("matched_domain", "Domain parsed from matched_website.", "string/blank"),

    ("country", "Country label for matched entity (best-effort).", "string/blank"),
    ("region", "Region derived from Wikidata continent (P30) via country.", "string/blank"),
    ("city_hq", "Headquarters location label (best-effort).", "string/blank"),
    ("industry", "Industry label from Wikidata (P452) if available.", "string/blank"),
    ("description_short", "Short description (best-effort).", "string/blank"),

    ("founded_year", "Founded year (YYYY) if available.", "string/blank"),
    ("employee_count", "Employee count if available.", "string/blank"),
    ("parent_company", "Parent company label if available.", "string/blank"),

    # Only keep stock_ticker (actual ticker text) + stock_exchange
    ("stock_ticker", "Stock ticker symbol (best-effort, via qualifier P249 on P414 statement).", "string/blank"),
    ("stock_exchange", "Stock exchange label (Wikidata P414) if available.", "string/blank"),

    ("field_coverage_score", "Share (0–1) of key enrichment fields filled for this row.", "float"),
    ("provenance_summary", "High-level provenance summary (MVP: 'wikidata' when matched).", "string"),
    ("row_last_updated", "UTC timestamp when row was generated.", "string"),
]


def write_data_dictionary(path: str) -> None:
    lines = []
    lines.append("# Output Data Dictionary (enriched.csv)\n\n")
    lines.append("| Column | Type | Description |\n")
    lines.append("|---|---|---|\n")
    for col, desc, typ in DATA_DICTIONARY:
        lines.append(f"| {col} | {typ} | {desc} |\n")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main() -> None:
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    input_csv = cfg["io"]["input_csv"]
    out_csv = cfg["io"]["enriched_csv"]
    out_audit = cfg["io"]["audit_jsonl"]
    out_summary = cfg["io"]["summary_md"]
    out_dict = "outputs/data_dictionary.md"

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_audit) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_summary) or ".", exist_ok=True)

    cache_cfg = cfg["cache"]
    cache = SimpleJsonCache(
        path=cache_cfg["path"],
        enabled=bool(cache_cfg["enabled"]),
        max_items=int(cache_cfg.get("max_items", 20000)),
    )

    wd_cfg = cfg["retrieval"]
    client = WikidataClient(
        user_agent=wd_cfg["user_agent"],
        throttle_seconds=float(wd_cfg["throttle_seconds"]),
        cache=cache,
    )

    m_cfg = cfg["matching"]
    mc = MatchConfig(
        w_name_fuzzy=float(m_cfg["weights"]["name_fuzzy"]),
        w_name_tfidf=float(m_cfg["weights"]["name_tfidf"]),
        w_domain=float(m_cfg["weights"]["domain_match"]),
        thr_high=float(m_cfg["thresholds"]["high"]),
        thr_medium=float(m_cfg["thresholds"]["medium"]),
        thr_low=float(m_cfg["thresholds"]["low"]),
    )

    prep = preprocess_input_csv(input_csv)
    df = prep.df

    t0 = time.time()
    print("Starting pipeline...", flush=True)

    enriched_rows = []
    stats = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "WEAK": 0, "NO_MATCH": 0}

    with open(out_audit, "w", encoding="utf-8") as audit_f:
        for i, (_, r) in enumerate(df.iterrows(), start=1):
            base_row = r.to_dict()
            if i % 50 == 0:
                print(f"Processed {i} rows in {time.time() - t0:.1f}s", flush=True)

            q = base_row.get("normalized_name") or ""
            domain = base_row.get("normalized_domain")

            domain_candidates = client.search_candidates_by_domain(domain, top_k=int(wd_cfg["top_k"])) if domain else []
            name_candidates = client.search_candidates(q, top_k=int(wd_cfg["top_k"]))

            if not name_candidates:
                for fq in _fallback_name_queries(q, base_row.get("input_company_name") or ""):
                    if fq.strip().lower() == (q or "").strip().lower():
                        continue
                    name_candidates = client.search_candidates(fq, top_k=int(wd_cfg["top_k"]))
                    if name_candidates:
                        break

            seen = set()
            candidates = []
            for c in domain_candidates + name_candidates:
                qid = c.get("entity_id")
                if not qid or qid in seen:
                    continue
                seen.add(qid)
                candidates.append(c)

            removed_by_org_filter = False
            if candidates:
                hinted_org = [c for c in candidates if _looks_like_org_by_text(c.get("label", ""), c.get("description", ""))]
                if hinted_org:
                    candidates = hinted_org
                else:
                    flags = client.orgish_flags([c.get("entity_id") for c in candidates if c.get("entity_id")])
                    if flags is not None:
                        org_only = [c for c in candidates if flags.get(c.get("entity_id"), False)]
                        if org_only:
                            candidates = org_only
                        else:
                            candidates = []
                            removed_by_org_filter = True

            if domain:
                for c in candidates:
                    if c.get("candidate_website"):
                        continue
                    qid = c.get("entity_id")
                    if qid:
                        mini = client.enrich_entity_minimal(qid)
                        c["candidate_website"] = mini.get("matched_website")

            scored = score_candidates(
                normalized_name=base_row.get("normalized_name", ""),
                normalized_domain=domain,
                candidates=candidates,
                cfg=mc,
            )

            if scored:
                top_conf = float(scored[0]["confidence"])
                status = classify_status(top_conf, mc)
                top_features = scored[0].get("features", {}) or {}
                reason = match_reason(top_features)
                top_qid = scored[0].get("entity_id")

                if status == "WEAK" and top_qid and top_features.get("domain_match", 0.0) >= 1.0:
                    is_orgish = True
                    try:
                        flags = client.orgish_flags([top_qid]) or {}
                        if flags:
                            is_orgish = bool(flags.get(top_qid, False))
                    except Exception:
                        pass
                    if is_orgish:
                        status = "LOW"
                        reason = "domain_match_override"

                enrichment = client.enrich_entity(top_qid) if (status != "NO_MATCH" and top_qid) else {}
            else:
                top_conf = 0.0
                status = "NO_MATCH"
                reason = "no_org_candidates" if removed_by_org_filter else "no_candidates"
                enrichment = {}

            enriched_row, audit = build_outputs(
                base_row=base_row,
                scored_candidates=scored,
                top_enrichment=enrichment,
                match_status=status,
                match_confidence=top_conf,
                match_reason_top=reason,
            )

            enriched_rows.append(enriched_row)
            stats[status] += 1
            audit_f.write(json.dumps(audit, ensure_ascii=False) + "\n")

    out_df = pd.DataFrame(enriched_rows)
    out_df.to_csv(out_csv, index=False)

    cache.save()

    total = len(out_df)
    avg_conf = float(out_df["match_confidence"].fillna(0).mean()) if total else 0.0
    avg_cov = float(out_df["field_coverage_score"].fillna(0).mean()) if total else 0.0

    cov_cols = [
        "matched_website",
        "matched_domain",
        "country",
        "region",
        "industry",
        "description_short",
        "founded_year",
        "employee_count",
        "parent_company",
        "stock_ticker",
        "stock_exchange",
        "city_hq",
    ]
    cov_lines = []
    for c in cov_cols:
        pct = 0.0 if total == 0 else (out_df[c].notna() & (out_df[c].astype(str).str.strip() != "")).mean()
        cov_lines.append(f"- {c}: {pct:.0%}")

    md = []
    md.append("# CIE MVP Run Summary\n\n")
    md.append(f"- Total rows: {total}\n")
    md.append(
        f"- HIGH: {stats['HIGH']} | MEDIUM: {stats['MEDIUM']} | LOW: {stats['LOW']} | WEAK: {stats['WEAK']} | NO_MATCH: {stats['NO_MATCH']}\n"
    )
    md.append(f"- Average match confidence: {avg_conf:.3f}\n")
    md.append(f"- Average field coverage score: {avg_cov:.3f}\n")
    md.append("\n## Field coverage\n")
    md.extend([line + "\n" for line in cov_lines])
    md.append("\n## Outputs\n")
    md.append(f"- Enriched CSV: `{out_csv}`\n")
    md.append(f"- Audit trail (JSONL): `{out_audit}`\n")
    md.append(f"- Data dictionary: `{out_dict}`\n")

    with open(out_summary, "w", encoding="utf-8") as f:
        f.writelines(md)

    write_data_dictionary(out_dict)

    print("✅ Done")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_audit}")
    print(f"Saved: {out_summary}")
    print(f"Saved: {out_dict}")


if __name__ == "__main__":
    main()