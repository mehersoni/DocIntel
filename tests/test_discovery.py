import pytest
from pathlib import Path
from app.discovery.merger import merge_discovered_categories
from app.discovery.statistical import extract_noun_phrases, run_statistical_discovery
from app.discovery.glossary_scan import run_direct_glossary_scan, extract_category_details

def test_merge_discovered_categories(tmp_path):
    statistical_cats = [
        {"name": "Equipment", "description": "Rig equipment"},
        {"name": "Incident", "description": "Rig safety incident"}
    ]
    llm_cats = [
        {"name": "Equipment", "description": "Detailed equipment description"},
        {"name": "Formation", "description": "Bakken geological formation"}
    ]
    
    output_file = tmp_path / "candidate_schema.yaml"
    merged = merge_discovered_categories(statistical_cats, llm_cats, output_file)
    
    # We expect 3 categories
    assert len(merged) == 3
    
    # Check Equipment (in both => HIGH)
    equipment_item = next(x for x in merged if x["name"] == "Equipment")
    assert equipment_item["confidence"] == "HIGH"
    assert "statistical" in equipment_item["sources"]
    assert "direct_scan" in equipment_item["sources"]
    assert equipment_item["description"] == "Detailed equipment description"
    
    # Check Incident (only in statistical => MEDIUM)
    incident_item = next(x for x in merged if x["name"] == "Incident")
    assert incident_item["confidence"] == "MEDIUM"
    assert incident_item["sources"] == ["statistical"]
    
    # Check Formation (only in LLM => MEDIUM)
    formation_item = next(x for x in merged if x["name"] == "Formation")
    assert formation_item["confidence"] == "MEDIUM"
    assert formation_item["sources"] == ["direct_scan"]

def test_extract_noun_phrases():
    chunks = [{"text": "The high-pressure mud motor failed during operations."}]
    phrases = extract_noun_phrases(chunks)
    
    assert len(phrases) > 0
    assert any("mud motor" in p for p in phrases)

def test_merge_details():
    from app.schema.manager import merge_details
    
    old_details = {
        "summary": "Existing summary preserved",
        "examples": ["Well A"],
        "common_attributes": ["Depth"],
        "observed_terms": ["wellbore"],
        "sample_mentions": ["...text..."]
    }
    
    new_details = {
        "summary": "New summary",
        "examples": ["Well A", "Well B"],
        "common_attributes": ["Depth", "Operator"],
        "observed_terms": ["wellbore", "casing"],
        "sample_mentions": ["...new text..."]
    }
    
    merged = merge_details(old_details, new_details)
    assert merged["summary"] == "Existing summary preserved"
    assert "Well B" in merged["examples"]
    assert "Operator" in merged["common_attributes"]
    assert "casing" in merged["observed_terms"]

def test_empty_and_malformed_document_handling():
    """Verifies that empty chunks or empty strings produce empty candidate sets cleanly without error."""
    empty_chunks = []
    res_stat = run_statistical_discovery(empty_chunks)
    assert res_stat == []
    
    blank_chunks = [{"doc_id": "test.txt", "chunk_index": 0, "text": "   "}]
    res_scan = run_direct_glossary_scan(blank_chunks)
    assert res_scan == []

def test_direct_glossary_scan_threshold():
    """Verifies that single term occurrences do not trigger false positive category matching."""
    # Single mention of a term (below threshold)
    single_chunks = [{"doc_id": "test.txt", "chunk_index": 0, "text": "A single packer was inspected."}]
    res_single = run_direct_glossary_scan(single_chunks)
    
    # Multiple mentions of terms (above threshold >= 2)
    multi_chunks = [
        {"doc_id": "test.txt", "chunk_index": 0, "text": "The hydraulic packer and production casing string were set at 3445 meters."},
        {"doc_id": "test.txt", "chunk_index": 1, "text": "Packer pressure reached 1850 psi during completion test."}
    ]
    res_multi = run_direct_glossary_scan(multi_chunks)
    assert len(res_multi) > 0

def test_confidence_merger_edge_cases(tmp_path):
    """Verifies edge cases in merger logic when input candidate lists are empty or overlap."""
    out_file = tmp_path / "edge_schema.yaml"
    merged_empty = merge_discovered_categories([], [], out_file)
    assert merged_empty == []
    
    merged_single = merge_discovered_categories([{"name": "Drilling", "description": "Drill"}], [], out_file)
    assert len(merged_single) == 1
    assert merged_single[0]["confidence"] == "MEDIUM"
