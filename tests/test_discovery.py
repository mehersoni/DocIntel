import pytest
from app.discovery.merger import merge_discovered_categories
from app.discovery.statistical import extract_noun_phrases

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
    # The noun phrase should be present in the extracted list
    assert any("mud motor" in p for p in phrases)

def test_merge_details():
    from app.schema.manager import merge_details
    
    old_details = {
        "summary": "Old summary",
        "examples": ["Well A"],
        "common_attributes": ["Depth"],
        "observed_terms": ["wellbore"],
        "sample_mentions": ["...text..."]
    }
    
    new_details = {
        "summary": "New summary",
        "examples": ["Well A", "Well B"],
        "common_attributes": ["Depth", "Operator"],
        "observed_terms": ["wellhead"],
        "sample_mentions": ["...another text..."]
    }
    
    merged = merge_details(old_details, new_details)
    
    # Check that old summary is kept (since it's not empty)
    assert merged["summary"] == "Old summary"
    # Check that lists are combined uniquely
    assert merged["examples"] == ["Well A", "Well B"]
    assert merged["common_attributes"] == ["Depth", "Operator"]
    assert merged["observed_terms"] == ["wellbore", "wellhead"]
    assert merged["sample_mentions"] == ["...text...", "...another text..."]

def test_save_approved_schema_with_details(tmp_path):
    import yaml
    from app.schema.manager import save_approved_schema
    
    approved_cats = [
        {
            "name": "Well",
            "description": "Oil well",
            "confidence": "HIGH",
            "sources": ["direct_scan"],
            "details": {
                "summary": "Well details summary",
                "examples": ["Well A"]
            }
        }
    ]
    
    edit_log = {"added": [], "removed": [], "renamed": {}}
    approved_path = tmp_path / "approved_schema.yaml"
    edit_log_path = tmp_path / "edit_log.json"
    
    save_approved_schema(approved_cats, edit_log, approved_path, edit_log_path)
    
    # Reload and check structure
    with open(approved_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    assert "categories" in data
    well_cat = data["categories"][0]
    assert well_cat["name"] == "Well"
    assert "details" in well_cat
    assert well_cat["details"]["summary"] == "Well details summary"
    assert well_cat["details"]["examples"] == ["Well A"]

def test_tfidf_statistical_discovery():
    from app.discovery.statistical import run_statistical_discovery
    chunks = [
        {"doc_id": "report1.pdf", "text": "Gamma ray log showed shale formations at depth of 8450 ft. Gamma ray logs evaluate lithology."},
        {"doc_id": "report1.pdf", "text": "Resistivity log indicated sandstone reservoir facies with high resistivity values. Density log and caliper log measured borehole diameter."}
    ]
    cats = run_statistical_discovery(chunks)
    assert isinstance(cats, list)
    cat_names = [c["name"] for c in cats]
    assert len(cat_names) > 0

def test_extract_category_details_numerical_attributes():
    from app.discovery.llm import extract_category_details
    chunks = [
        {"doc_id": "doc1.pdf", "text": "The Bakken shale target depth was 8,450 - 9,200 ft with mud weight 12.5 ppg and pressure 4500 psi."}
    ]
    details = extract_category_details("Geology", "Geological formations", chunks)
    assert "numerical_attributes" in details
    num_attrs = details["numerical_attributes"]
    assert len(num_attrs) > 0
    assert any("8,450" in a or "12.5 ppg" in a or "4500 psi" in a for a in num_attrs)
