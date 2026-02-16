"""
Company Information Enrichment (CIE) Tool - MVP package.

Keeps src/ importable so run_pipeline.py can do:
from src.preprocess import ...
"""
__all__ = [
    "preprocess",
    "wikidata_client",
    "matching",
    "enrichment",
    "utils",
]