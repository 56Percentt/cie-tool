from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import pandas as pd

from .utils import safe_str


LEGAL_SUFFIXES = [
    "ltd", "limited", "inc", "inc.", "corp", "corporation", "co", "co.", "company",
    "plc", "llc", "gmbh", "ag", "sa", "s.a.", "bv", "pte", "pvt", "sarl",
    "holdings", "group"  # optional: helps reduce noise; comment out if you dislike
]

_SUFFIX_RE = re.compile(r"\b(" + "|".join(re.escape(s) for s in LEGAL_SUFFIXES) + r")\b", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def normalize_company_name(name: str) -> str:
    s = safe_str(name).lower()
    if not s:
        return ""

    s = re.sub(r"[&/]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = _SUFFIX_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def clean_website(website: Optional[str]) -> Optional[str]:
    w = safe_str(website)
    if not w:
        return None

    if "://" not in w:
        w = "https://" + w

    try:
        parsed = urlparse(w)
        if not parsed.netloc:
            return None
        return f"{parsed.scheme}://{parsed.netloc}".lower()
    except Exception:
        return None


def extract_domain(cleaned_website: Optional[str]) -> Optional[str]:
    if not cleaned_website:
        return None
    try:
        parsed = urlparse(cleaned_website)
        d = (parsed.netloc or "").lower().strip(".")
        return d or None
    except Exception:
        return None


@dataclass
class PreprocessResult:
    df: pd.DataFrame
    n_rows: int
    n_missing_website: int
    n_missing_company_name: int


def preprocess_input_csv(input_path: str) -> PreprocessResult:
    df = pd.read_csv(input_path)

    if "company_name" not in df.columns:
        raise ValueError("Input CSV must contain 'company_name' column.")
    if "website" not in df.columns:
        df["website"] = None  # optional

    df["input_company_name"] = df["company_name"]
    df["input_website"] = df["website"]

    df["company_name"] = df["company_name"].apply(safe_str)
    df["website"] = df["website"].apply(clean_website)

    df["normalized_name"] = df["company_name"].apply(normalize_company_name)
    df["normalized_domain"] = df["website"].apply(extract_domain)

    n_rows = len(df)
    n_missing_website = int(df["website"].isna().sum())
    n_missing_company_name = int(df["normalized_name"].eq("").sum())

    keep_cols = [
        "input_company_name", "input_website",
        "company_name", "website",
        "normalized_name", "normalized_domain",
    ]
    out = df[keep_cols].copy()

    return PreprocessResult(
        df=out,
        n_rows=n_rows,
        n_missing_website=n_missing_website,
        n_missing_company_name=n_missing_company_name,
    )