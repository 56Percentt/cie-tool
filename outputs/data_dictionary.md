# Output Data Dictionary (enriched.csv)

| Column | Type | Description |
|---|---|---|
| input_company_name | string | Original company name from input CSV. |
| input_website | string/blank | Original website from input CSV (may be blank). |
| normalized_name | string | Normalized company name used for retrieval/matching. |
| normalized_domain | string/blank | Domain extracted from input website (if provided). |
| match_status | categorical | Confidence bucket: HIGH/MEDIUM/LOW/WEAK. NO_MATCH means no viable candidate was found. |
| match_confidence | float | Probability-like match score (0–1). |
| match_reason_top | string | Reason label for top match (domain_match/strong_name_match/ambiguous_name). |
| matched_entity_id | string/blank | Wikidata entity id (QID) for selected match. |
| matched_official_name | string/blank | Best available official name for matched entity. |
| matched_website | string/blank | Matched entity website (best-effort). |
| matched_domain | string/blank | Domain parsed from matched_website. |
| country | string/blank | Country label for matched entity (best-effort). |
| region | string/blank | Region derived from Wikidata continent (P30) via country. |
| city_hq | string/blank | Headquarters location label (best-effort). |
| industry | string/blank | Industry label from Wikidata (P452) if available. |
| description_short | string/blank | Short description (best-effort). |
| founded_year | string/blank | Founded year (YYYY) if available. |
| employee_count | string/blank | Employee count if available. |
| parent_company | string/blank | Parent company label if available. |
| stock_ticker | string/blank | Stock ticker symbol (best-effort, via qualifier P249 on P414 statement). |
| stock_exchange | string/blank | Stock exchange label (Wikidata P414) if available. |
| field_coverage_score | float | Share (0–1) of key enrichment fields filled for this row. |
| provenance_summary | string | High-level provenance summary (MVP: 'wikidata' when matched). |
| row_last_updated | string | UTC timestamp when row was generated. |
