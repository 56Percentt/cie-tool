# CIE Tool (Company Identity Enrichment)

CIE Tool enriches company names (optionally with domains) by matching them to Wikidata entities and outputting an enriched CSV with confidence scores and match metadata. It includes a Streamlit demo webapp for single lookups or batch CSV uploads.

Live demo: [https://cie-tool.streamlit.app](https://cie-tool.streamlit.app)

---

## What it does

Given a company name (and optionally a domain/website), the tool:

* searches Wikidata for candidate entities
* filters/ranks candidates and selects the best match
* assigns a match status + confidence score
* returns an enriched CSV with additional fields (e.g., matched entity id/name/domain, location, industry, etc.)

---

## Demo flow (Streamlit app)

* Enter a company name (optional domain), OR upload a CSV of companies
* Click “Run enrichment”
* View results in the app + download the enriched CSV
* Optional: view the progress log and choose which enriched fields to include in the output

---

## CSV input format

Your CSV must include a company name column. Supported column names:

Company name (required):

* company_name (preferred)
* company
* name

Domain / website (optional):

* domain
* website
* url

Example:

```
company_name,website
Stripe,stripe.com
OpenAI,openai.com
```

---

## Run locally

1. Install dependencies

   pip install -r requirements.txt

2. Launch the demo app

   streamlit run streamlit_app.py

---

## Repository structure

* streamlit_app.py — Streamlit demo webapp
* run_pipeline.py — CLI pipeline runner
* src/ — core enrichment logic
* config.yaml — configuration

---

## Notes

* This project queries Wikidata; please be polite with request rates (the app includes throttling and caching).
* Avoid committing generated outputs/caches (see .gitignore).

---

## License

MIT License. See LICENSE.

---
Copy/paste this entire thing into `README.md` (it’s plain Markdown, no outer code fence):

---

# CIE Tool (Company Identity Enrichment)

CIE Tool enriches company names (optionally with domains) by matching them to Wikidata entities and outputting an enriched CSV with confidence scores and match metadata. It includes a Streamlit demo webapp for single lookups or batch CSV uploads.

Live demo: [https://cie-tool.streamlit.app](https://cie-tool.streamlit.app)

---

## What it does

Given a company name (and optionally a domain/website), the tool:

* searches Wikidata for candidate entities
* filters/ranks candidates and selects the best match
* assigns a match status + confidence score
* returns an enriched CSV with additional fields (e.g., matched entity id/name/domain, location, industry, etc.)

---

## Demo flow (Streamlit app)

* Enter a company name (optional domain), OR upload a CSV of companies
* Click “Run enrichment”
* View results in the app + download the enriched CSV
* Optional: view the progress log and choose which enriched fields to include in the output

---

## CSV input format

Your CSV must include a company name column. Supported column names:

Company name (required):

* company_name (preferred)
* company
* name

Domain / website (optional):

* domain
* website
* url

Example:

```
company_name,website
Stripe,stripe.com
OpenAI,openai.com
```

---

## Run locally

1. Install dependencies

   pip install -r requirements.txt

2. Launch the demo app

   streamlit run streamlit_app.py

---

## Repository structure

* streamlit_app.py — Streamlit demo webapp
* run_pipeline.py — CLI pipeline runner
* src/ — core enrichment logic
* config.yaml — configuration

---

## Notes

* This project queries Wikidata; please be polite with request rates (the app includes throttling and caching).
* Avoid committing generated outputs/caches (see .gitignore).

---

## License

MIT License. See LICENSE.

---
