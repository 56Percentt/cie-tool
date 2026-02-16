from __future__ import annotations

import io
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
import streamlit as st

from src.enrichment import build_outputs
from src.matching import MatchConfig, classify_status, match_reason, score_candidates
from src.preprocess import clean_website, extract_domain, normalize_company_name
from src.utils import SimpleJsonCache, safe_str
from src.wikidata_client import WikidataClient


# -----------------------------
# Small helpers copied from run_pipeline.py
# -----------------------------
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


# -----------------------------
# Streamlit app
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE_PATH = str(APP_DIR / ".cache_wikidata.json")

# Default (recommended) enriched fields to include in the downloadable CSV.
# Anything not present in the actual output columns will be ignored.
RECOMMENDED_FIELDS = [
    "match_status",
    "match_confidence",
    "match_reason_top",
    "matched_entity_id",
    "matched_official_name",
    "matched_domain",
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


def _infer_cols(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """
    Try to infer company name + website/domain columns.
    Returns: (company_col, website_col_or_none)
    """
    cols = {c.lower().strip(): c for c in df.columns}

    company_candidates = ["company_name", "company", "name", "account_name"]
    website_candidates = ["website", "domain", "url", "homepage", "company_domain"]

    company_col = None
    for k in company_candidates:
        if k in cols:
            company_col = cols[k]
            break
    if company_col is None:
        raise ValueError("Couldn't find a company name column. Please include 'company_name' (recommended).")

    website_col = None
    for k in website_candidates:
        if k in cols:
            website_col = cols[k]
            break

    return company_col, website_col


def _prep_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the minimal columns required by the pipeline.
    Keeps the row order.
    """
    company_col, website_col = _infer_cols(df_raw)

    out = pd.DataFrame()
    out["input_company_name"] = df_raw[company_col].apply(safe_str)

    if website_col is None:
        out["input_website"] = None
    else:
        out["input_website"] = df_raw[website_col].apply(safe_str)

    out["company_name"] = out["input_company_name"].apply(safe_str)
    out["website"] = out["input_website"].apply(clean_website)
    out["normalized_name"] = out["company_name"].apply(normalize_company_name)
    out["normalized_domain"] = out["website"].apply(extract_domain)
    return out


@st.cache_resource(show_spinner=False)
def _get_client(user_agent: str, throttle_seconds: float, cache_enabled: bool, cache_path: str) -> Tuple[WikidataClient, SimpleJsonCache]:
    cache = SimpleJsonCache(path=cache_path, enabled=cache_enabled, max_items=20000)
    client = WikidataClient(
        user_agent=user_agent,
        throttle_seconds=float(throttle_seconds),
        cache=cache,
    )
    return client, cache


def _get_match_config() -> MatchConfig:
    # Mirrors config.yaml defaults (you can expose these as sliders if you want).
    return MatchConfig(
        w_name_fuzzy=0.45,
        w_name_tfidf=0.35,
        w_domain=0.20,
        thr_high=0.80,
        thr_medium=0.60,
        thr_low=0.45,
    )


def run_cie(
    prep_df: pd.DataFrame,
    client: WikidataClient,
    cache: SimpleJsonCache,
    top_k: int,
    log_cb: Optional[Callable[[str], None]] = None,
    log_every: int = 1,
) -> Tuple[pd.DataFrame, list[dict]]:
    mc = _get_match_config()

    enriched_rows: list[dict] = []
    audits: list[dict] = []

    progress = st.progress(0)
    status = st.empty()

    t0 = time.time()
    total = len(prep_df)

    log_every = max(int(log_every), 1)

    if log_cb:
        log_cb(f"Starting enrichment for {total} row(s)...")

    for i, (_, r) in enumerate(prep_df.iterrows(), start=1):
        base_row = r.to_dict()

        q = base_row.get("normalized_name") or ""
        domain = base_row.get("normalized_domain") or None

        if log_cb and (i == 1 or i % log_every == 0 or i == total):
            company_disp = safe_str(base_row.get("input_company_name"))
            domain_disp = safe_str(domain)
            log_cb(f"[{i}/{total}] {company_disp} • domain={domain_disp or '-'}")

        # 1) candidates by domain (if provided) + by name
        domain_candidates = client.search_candidates_by_domain(domain, top_k=top_k) if domain else []
        name_candidates = client.search_candidates(q, top_k=top_k)

        if not name_candidates:
            for fq in _fallback_name_queries(q, base_row.get("input_company_name") or ""):
                if fq.strip().lower() == (q or "").strip().lower():
                    continue
                name_candidates = client.search_candidates(fq, top_k=top_k)
                if name_candidates:
                    if log_cb:
                        log_cb(f"    fallback name query: {fq}")
                    break

        # 2) dedupe candidates
        seen = set()
        candidates = []
        for c in domain_candidates + name_candidates:
            qid = c.get("entity_id")
            if not qid or qid in seen:
                continue
            seen.add(qid)
            candidates.append(c)

        # 3) filter to "org-ish" entities (helps avoid fruit / units / etc.)
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

        if log_cb and (i == 1 or i % log_every == 0 or i == total):
            log_cb(f"    candidates after org filter: {len(candidates)}")

        # 4) minimal enrichment for candidates (website) when domain is provided
        if domain:
            for c in candidates:
                if c.get("candidate_website"):
                    continue
                qid = c.get("entity_id")
                if qid:
                    mini = client.enrich_entity_minimal(qid)
                    c["candidate_website"] = mini.get("matched_website")

        # 5) score + classify + enrich top match
        scored = score_candidates(
            normalized_name=base_row.get("normalized_name", ""),
            normalized_domain=domain,
            candidates=candidates,
            cfg=mc,
        )

        if scored:
            top_conf = float(scored[0]["confidence"])
            match_status = classify_status(top_conf, mc)
            top_features = scored[0].get("features", {}) or {}
            reason = match_reason(top_features)
            top_qid = scored[0].get("entity_id")

            # domain-match override (same logic as run_pipeline.py)
            if match_status == "WEAK" and top_qid and top_features.get("domain_match", 0.0) >= 1.0:
                is_orgish = True
                try:
                    flags = client.orgish_flags([top_qid]) or {}
                    if flags:
                        is_orgish = bool(flags.get(top_qid, False))
                except Exception:
                    pass
                if is_orgish:
                    match_status = "LOW"
                    reason = "domain_match_override"

            top_enrichment = client.enrich_entity(top_qid) if (match_status != "NO_MATCH" and top_qid) else {}
        else:
            top_conf = 0.0
            match_status = "NO_MATCH"
            reason = "no_org_candidates" if removed_by_org_filter else "no_candidates"
            top_enrichment = {}

        if log_cb and (i == 1 or i % log_every == 0 or i == total):
            log_cb(f"    → {match_status} conf={top_conf:.3f} reason={reason}")

        enriched_row, audit = build_outputs(
            base_row=base_row,
            scored_candidates=scored,
            top_enrichment=top_enrichment,
            match_status=match_status,
            match_confidence=top_conf,
            match_reason_top=reason,
        )

        enriched_rows.append(enriched_row)
        audits.append(audit)

        # UI updates
        if total:
            progress.progress(min(i / total, 1.0))
        if i == 1 or i % 10 == 0 or i == total:
            status.markdown(f"**Processed {i}/{total}** • elapsed {time.time() - t0:.1f}s")

    if log_cb:
        log_cb("Saving cache…")
    cache.save()
    status.empty()
    progress.empty()

    if log_cb:
        log_cb("Done ✅")

    return pd.DataFrame(enriched_rows), audits

def _download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def main() -> None:
    st.set_page_config(page_title="CIE Demo", layout="wide")
    st.title("CIE Demo — Company Intelligence Enrichment")
    st.caption("Enter one company or upload a CSV, then run the enrichment pipeline and download an enriched CSV.")

    # Persist results/logs across widget changes (so changing column selection doesn't re-run Wikidata calls).
    st.session_state.setdefault("cie_result", None)
    st.session_state.setdefault("cie_log_lines", [])
    st.session_state.setdefault("selected_enriched_cols", [])

    with st.sidebar:
        st.header("Input")
        mode = st.radio("Choose input method", ["Single company", "Upload CSV"], horizontal=False)

        st.divider()
        st.header("Run settings")

        # NOTE: Wikidata requests should include a descriptive User-Agent.
        default_ua = os.environ.get("CIE_USER_AGENT", "CIE-Demo/1.0 (contact: you@example.com)")
        user_agent = st.text_input("User-Agent", value=default_ua, help="Please include a contact email for Wikidata politeness.")

        throttle = st.slider("Throttle (seconds between requests)", 0.0, 2.0, 0.2, 0.1)
        top_k = st.slider("Candidates to retrieve (top_k)", 1, 15, 5, 1)
        log_every = st.slider("Log every N rows", 1, 50, 1, 1, help="Higher values reduce UI churn for large CSVs.")


        cache_enabled = st.checkbox("Enable on-disk cache", value=True)
        cache_path = st.text_input("Cache path", value=DEFAULT_CACHE_PATH)

        st.divider()
        st.header("Safety limits")
        max_rows = st.number_input("Max rows per run", min_value=1, max_value=5000, value=250, step=25)

        st.divider()
        if st.button("Clear Streamlit cache"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.toast("Cleared Streamlit cache.", icon="🧹")
        if st.button("Delete Wikidata cache file"):
            try:
                Path(cache_path).unlink(missing_ok=True)
                st.toast("Deleted cache file.", icon="🗑️")
            except Exception as e:
                st.warning(f"Couldn't delete cache file: {e}")

    df_raw: Optional[pd.DataFrame] = None

    if mode == "Single company":
        c1, c2 = st.columns([2, 2])
        with c1:
            company = st.text_input("Company name", placeholder="e.g., Stripe")
        with c2:
            domain = st.text_input("Domain / website (optional)", placeholder="e.g., stripe.com")

        if company.strip():
            df_raw = pd.DataFrame([{"company_name": company.strip(), "website": domain.strip() if domain.strip() else None}])
            st.info("Tip: For best results, provide the company website/domain when you have it.")
    else:
        up = st.file_uploader("Upload a CSV", type=["csv"])
        if up is not None:
            df_raw = pd.read_csv(up)

            st.write("Preview of uploaded CSV:")
            st.dataframe(df_raw.head(25), use_container_width=True)

            st.caption("Your CSV should include a company name column (recommended: `company_name`) and optionally `website` or `domain`.")

    can_run = df_raw is not None and len(df_raw) > 0

    # Progress log panel (persisted in session_state)
    log_expander = st.expander("Progress log", expanded=True)
    with log_expander:
        log_box = st.empty()

        # Render current log
        if st.session_state.cie_log_lines:
            log_box.code("\n".join(st.session_state.cie_log_lines[-200:]), language="text")

        c_log1, c_log2 = st.columns([1, 1])
        with c_log1:
            if st.button("Clear log", disabled=not bool(st.session_state.cie_log_lines)):
                st.session_state.cie_log_lines = []
                log_box.empty()
        with c_log2:
            st.download_button(
                "Download log",
                data="\n".join(st.session_state.cie_log_lines).encode("utf-8"),
                file_name="cie_run_log.txt",
                mime="text/plain",
                disabled=not bool(st.session_state.cie_log_lines),
            )

    def _log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        st.session_state.cie_log_lines.append(f"{ts} {msg}")
        st.session_state.cie_log_lines = st.session_state.cie_log_lines[-2000:]
        log_box.code("\n".join(st.session_state.cie_log_lines[-200:]), language="text")

    run = st.button("▶ Run enrichment", type="primary", disabled=not can_run)

    if run and df_raw is not None:
        # New run: clear old log + old result
        st.session_state.cie_log_lines = []
        st.session_state.cie_result = None
        st.session_state.selected_enriched_cols = []

        if len(df_raw) > int(max_rows):
            st.error(f"CSV has {len(df_raw)} rows, which exceeds the demo limit ({max_rows}). Reduce rows or raise the limit in the sidebar.")
            st.stop()

        try:
            prep = _prep_input(df_raw)
        except Exception as e:
            st.error(str(e))
            st.stop()

        client, cache = _get_client(
            user_agent=user_agent,
            throttle_seconds=float(throttle),
            cache_enabled=bool(cache_enabled),
            cache_path=str(cache_path),
        )

        with st.spinner("Running enrichment..."):
            enriched_df, audits = run_cie(
                prep,
                client=client,
                cache=cache,
                top_k=int(top_k),
                log_cb=_log,
                log_every=int(log_every),
            )

        # Persist full results in session_state so column selection doesn't re-run enrichment.
        st.session_state.cie_result = {
            "df_raw": df_raw,
            "enriched_df": enriched_df,
            "audits": audits,
            "ran_at": time.time(),
        }

    # ---- Display last results (if any) ----
    result = st.session_state.cie_result
    if result is not None:
        df_raw_res: pd.DataFrame = result["df_raw"]
        enriched_df: pd.DataFrame = result["enriched_df"]
        audits: list[dict] = result["audits"]

        # Merge back to original (keep original columns intact)
        new_cols = [c for c in enriched_df.columns if c not in df_raw_res.columns]
        merged_full = pd.concat([df_raw_res.reset_index(drop=True), enriched_df[new_cols].reset_index(drop=True)], axis=1)

        st.subheader("Run summary")
        if "match_status" in merged_full.columns:
            counts = merged_full["match_status"].value_counts(dropna=False).to_dict()
            cols = st.columns(5)
            for j, k in enumerate(["HIGH", "MEDIUM", "LOW", "WEAK", "NO_MATCH"]):
                cols[j].metric(k, int(counts.get(k, 0)))
        st.caption(
            f"Rows: {len(merged_full)} • Avg confidence: {float(merged_full.get('match_confidence', pd.Series([0])).fillna(0).mean()):.3f}"
        )

        # Column selection for output
        st.subheader("Output columns")

        if not new_cols:
            st.info("No enriched columns were produced for this run.")
            st.stop()

        recommended = [c for c in RECOMMENDED_FIELDS if c in new_cols]

        # Keep selection valid and initialize default selection once per run
        st.session_state.selected_enriched_cols = [c for c in st.session_state.selected_enriched_cols if c in new_cols]
        if not st.session_state.selected_enriched_cols:
            st.session_state.selected_enriched_cols = recommended or new_cols

        b1, b2, b3 = st.columns([1, 1, 1])
        with b1:
            if st.button("Use recommended", disabled=not bool(recommended)):
                st.session_state.selected_enriched_cols = recommended
        with b2:
            if st.button("Use all"):
                st.session_state.selected_enriched_cols = new_cols
        with b3:
            if st.button("Clear last result"):
                st.session_state.cie_result = None
                st.session_state.selected_enriched_cols = []
                st.rerun()

        selected_cols = st.multiselect(
            "Choose enriched fields to include in the downloadable CSV",
            options=new_cols,
            key="selected_enriched_cols",
        )

        merged_selected = pd.concat(
            [df_raw_res.reset_index(drop=True), enriched_df[selected_cols].reset_index(drop=True)],
            axis=1,
        )

        st.subheader("Enriched CSV")
        st.dataframe(merged_selected, use_container_width=True, height=520)

        st.download_button(
            "⬇ Download enriched CSV",
            data=_download_bytes(merged_selected),
            file_name="cie_enriched.csv",
            mime="text/csv",
        )

        with st.expander("Audit trail (first 3 rows)"):
            st.json(audits[:3])


if __name__ == "__main__":
    main()