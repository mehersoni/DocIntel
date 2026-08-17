# Compatibility shim for legacy imports
from app.discovery.glossary_scan import run_direct_glossary_scan, extract_category_details, select_representative_chunks

# Alias run_llm_discovery to run_direct_glossary_scan
run_llm_discovery = run_direct_glossary_scan
