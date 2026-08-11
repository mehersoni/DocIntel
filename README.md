# DocIntel: Human-in-the-Loop Schema Discovery Platform

DocIntel is an interactive, human-in-the-loop schema discovery platform for unstructured technical reports (PDF, DOCX, TXT). It extracts candidate entity categories using a 100% local, offline hybrid approach (spaCy noun phrase analysis + TF-IDF term weighting + direct domain glossary dictionary matching), extracts numerical measurements and document images, renders interactive co-occurrence heatmaps and 2D visual knowledge graphs, and provides a streamlined UI to review, rename, approve, or export schemas as YAML with audit edit logs.

No external APIs, cloud services, or API keys are required. All computation runs 100% locally and privately.

---

## Key Features

1. **Document Ingestion & Image Extraction**: Parses PDF, DOCX, and TXT reports, normalizes whitespace, extracts embedded figures and illustrations, and partitions text into ~500-word chunks.
2. **Statistical Discovery & TF-IDF Weighting**: Extracts noun phrases using spaCy, computes TF-IDF term weighting across chunks, and matches terms against a local technical glossary dictionary.
3. **Direct Glossary Scanning**: Scans document chunks for occurrences of specialized domain terms.
4. **Hybrid Category Merge**: Merges discovery outputs into candidate categories with confidence scoring (HIGH vs. MEDIUM).
5. **Numerical Parameter Extraction**: Uses regex patterns to identify quantitative measurements (depths, pressures, flow rates, fluid densities, tubular dimensions).
6. **Interactive Visual Analytics**: Generates Plotly co-occurrence matrix heatmaps and interactive Vis.js 2D visual knowledge graphs.
7. **Human-in-the-Loop Review**: Streamlined 3-subtab workspace to review, edit, approve, or reject categories, add custom categories, edit terms, and view sentence context.
8. **Schema Approval & Export**: Exports `approved_schema.yaml`, audit `edit_log.json`, and downloadable discovery payloads.

---

## Directory Structure

```text
docintel/
├── app/
│   ├── discovery/
│   │   ├── llm.py             # Direct glossary scanning & details extraction
│   │   ├── merger.py          # Merges Statistical & Direct scan candidates
│   │   └── statistical.py     # spaCy parsing + TF-IDF offline glossary matching
│   ├── ingestion/
│   │   ├── chunker.py         # Chunks text into ~500-word partitions
│   │   └── extractor.py       # PDF, DOCX, TXT text & image extraction
│   ├── schema/
│   │   └── manager.py         # Manages schema YAML, image metadata, and edit logs
│   ├── ui/
│   │   └── main.py            # Streamlit dashboard layout & interactive sub-tabs
│   └── config.py              # Configuration settings and defaults
├── data/
│   ├── sample_reports/        # Technical sample reports for demoing
│   └── oilfield_glossary.json # Local database of technical terms across disciplines
├── tests/                     # Automated unit test suite
├── requirements.txt           # Python dependency requirements
└── README.md                  # Setup & execution instructions
```

---

## Setup Instructions

### Prerequisites
- Python 3.11 or higher

### Installation

1. **Set up Virtual Environment and Install Dependencies**:
   Using `uv` (recommended):
   ```bash
   uv venv
   .venv\Scripts\activate      # On Windows
   uv pip install -r requirements.txt
   ```
   Or standard `pip`:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # On Windows
   pip install -r requirements.txt
   ```

2. **Install the spaCy Model** (Offline):
   ```bash
   uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
   ```

---

## How to Run

1. **Start the Streamlit App**:
   ```bash
   streamlit run app/ui/main.py
   ```
2. **Access the Dashboard**:
   Open [http://127.0.0.1:8501](http://127.0.0.1:8501) in your browser.

---

## Running Automated Tests

Run the pytest suite to verify all ingestion, discovery, and schema management components:
```bash
pytest
```

---

## Expected Outputs

All outputs are saved to the `outputs/` directory and can be downloaded directly from the UI:
- `outputs/chunks.json`: Normalized and chunked document texts.
- `outputs/schema_discovery_input.json`: Sub-sample of chunks analyzed during direct scanning.
- `outputs/candidate_schema.yaml`: Merged candidate categories before review.
- `outputs/approved_schema.yaml`: Finalized and approved categories with nested details, illustrations, and images.
- `outputs/edit_log.json`: Audit log of user modifications (added, removed, renamed).
