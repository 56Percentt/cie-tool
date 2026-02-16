from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .utils import now_iso


def extract_domain_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    u = str(url).strip()
    if not u:
        return None
    if "://" not in u:
        u = "https://" + u
    try:
        return (urlparse(u).netloc or "").lower().strip(".") or None
    except Exception:
        return None


def map_industry_to_taxonomy(industry_raw: Optional[str], desc: Optional[str], instance_of: Optional[str]) -> str:
    text = " ".join([str(x or "") for x in [industry_raw, desc, instance_of]]).lower()

    if any(
        k in text
        for k in [
            "non-governmental",
            "ngo",
            "nonprofit",
            "charity",
            "humanitarian",
            "development organisation",
            "development organization",
        ]
    ):
        return "NGO / Nonprofit"

    if any(
        k in text
        for k in [
            "consumer goods",
            "fmcg",
            "soap",
            "detergent",
            "toilet preparations",
            "personal care",
            "household products",
            "food",
            "beverage",
            "cosmetics",
        ]
    ):
        return "FMCG / Consumer Goods"

    if any(
        k in text
        for k in [
            "technology",
            "software",
            "internet",
            "search engine",
            "cloud computing",
            "artificial intelligence",
            "ai",
            "platform",
            "online",
            "computer",
        ]
    ):
        return "Technology"

    if any(k in text for k in ["bank", "insurance", "financial services", "microfinance", "fintech"]):
        return "Financial Services"

    if any(k in text for k in ["telecommunications", "mobile network", "carrier", "telco"]):
        return "Telecommunications"

    if any(k in text for k in ["healthcare", "medical", "hospital", "pharmaceutical"]):
        return "Healthcare"

    if any(k in text for k in ["university", "school", "education", "educational"]):
        return "Education"

    if "manufactur" in text:
        return "Manufacturing"

    return "Other"


def field_coverage_score(row: Dict[str, Any], fields: List[str]) -> float:
    filled = 0
    for f in fields:
        v = row.get(f)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        filled += 1
    return filled / max(1, len(fields))


def per_field_confidence(match_conf: float, value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, str) and not value.strip():
        return 0.0
    return float(match_conf)


def build_outputs(
    base_row: Dict[str, Any],
    scored_candidates: List[Dict[str, Any]],
    top_enrichment: Dict[str, Any],
    match_status: str,
    match_confidence: float,
    match_reason_top: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    top = scored_candidates[0] if scored_candidates else {}

    matched_website = top_enrichment.get("matched_website") or top.get("candidate_website")
    matched_domain = extract_domain_from_url(matched_website)

    desc = top_enrichment.get("description") or top.get("description") or None
    country = top_enrichment.get("country")
    region = top_enrichment.get("continent")
    industry = top_enrichment.get("industry_raw")

    enriched = {
        "input_company_name": base_row.get("input_company_name"),
        "input_website": base_row.get("input_website"),
        "normalized_name": base_row.get("normalized_name"),
        "normalized_domain": base_row.get("normalized_domain"),

        "match_status": match_status,
        "match_confidence": match_confidence,
        "match_reason_top": match_reason_top,
        "matched_entity_id": top.get("entity_id"),
        "matched_official_name": top_enrichment.get("matched_official_name") or top.get("label"),
        "matched_website": matched_website,
        "matched_domain": matched_domain,

        "country": country,
        "region": region,
        "city_hq": top_enrichment.get("city_hq"),
        "industry": industry,
        "description_short": desc,

        "founded_year": top_enrichment.get("founded_year"),
        "employee_count": top_enrichment.get("employee_count"),
        "parent_company": top_enrichment.get("parent_company"),

        # Keep only stock_ticker (no ticker_symbol column)
        "stock_ticker": top_enrichment.get("stock_ticker"),
        "stock_exchange": top_enrichment.get("stock_exchange"),

        "field_coverage_score": None,
        "provenance_summary": "wikidata" if top.get("entity_id") else "",
        "row_last_updated": now_iso(),
    }

    coverage_fields = [
        "matched_official_name",
        "matched_website",
        "country",
        "region",
        "city_hq",
        "industry",
        "description_short",
        "founded_year",
        "employee_count",
        "parent_company",
        "stock_ticker",
        "stock_exchange",
    ]
    enriched["field_coverage_score"] = field_coverage_score(enriched, coverage_fields)

    provenance = (top_enrichment.get("provenance") or {})
    audit = {
        "input": {
            "company_name": base_row.get("input_company_name"),
            "website": base_row.get("input_website"),
            "normalized_name": base_row.get("normalized_name"),
            "normalized_domain": base_row.get("normalized_domain"),
        },
        "match": {
            "status": match_status,
            "confidence": match_confidence,
            "reason_top": match_reason_top,
            "top_candidate": {
                "entity_id": top.get("entity_id"),
                "label": top.get("label"),
                "description": top.get("description"),
                "source": top.get("source"),
                "features": top.get("features", {}),
                "score": top.get("score"),
                "candidate_website": top.get("candidate_website"),
            },
            "alternatives": [
                {
                    "entity_id": c.get("entity_id"),
                    "label": c.get("label"),
                    "description": c.get("description"),
                    "source": c.get("source"),
                    "features": c.get("features", {}),
                    "score": c.get("score"),
                    "candidate_website": c.get("candidate_website"),
                }
                for c in scored_candidates[1:3]
            ],
        },
        "fields": {},
        "generated_at": now_iso(),
    }

    field_pairs = [
        ("matched_official_name", enriched.get("matched_official_name")),
        ("matched_website", enriched.get("matched_website")),
        ("matched_domain", enriched.get("matched_domain")),
        ("country", enriched.get("country")),
        ("region", enriched.get("region")),
        ("city_hq", enriched.get("city_hq")),
        ("industry", enriched.get("industry")),
        ("description_short", enriched.get("description_short")),
        ("founded_year", enriched.get("founded_year")),
        ("employee_count", enriched.get("employee_count")),
        ("parent_company", enriched.get("parent_company")),
        ("stock_ticker", enriched.get("stock_ticker")),
        ("stock_exchange", enriched.get("stock_exchange")),
    ]

    for field_key, value in field_pairs:
        if field_key == "matched_domain" and value:
            sources = ["derived_from_matched_website"]
        elif field_key == "industry" and value:
            sources = provenance.get("industry_raw", ["wikidata"])
        else:
            sources = provenance.get(field_key, ["wikidata"]) if value else []

        audit["fields"][field_key] = {
            "value": value,
            "sources": sources,
            "confidence": per_field_confidence(match_confidence, value),
        }

    return enriched, audit