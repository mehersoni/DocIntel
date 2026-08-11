import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
from app.config import APPROVED_SCHEMA_PATH, EDIT_LOG_PATH

def load_yaml_schema(file_path: Path) -> List[Dict[str, Any]]:
    """
    Loads categories from a YAML schema file.
    Returns an empty list if file doesn't exist or is invalid.
    """
    if not file_path.exists():
        return []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict) and "categories" in data:
                return data["categories"]
    except Exception:
        pass
    return []

def save_approved_schema(
    approved_categories: List[Dict[str, Any]],
    edit_log: Dict[str, Any],
    approved_path: Path = APPROVED_SCHEMA_PATH,
    edit_log_path: Path = EDIT_LOG_PATH,
    document_summary: str = "",
    document_captions: List[str] = None,
    extracted_images: List[Dict[str, Any]] = None
) -> None:
    """
    Saves approved categories, along with document metadata, to approved_schema.yaml,
    and writes the edit log to edit_log.json.
    """
    # Write approved categories to approved_schema.yaml
    schema_data = {
        "categories": approved_categories
    }
    if document_summary:
        schema_data["document_summary"] = document_summary
    if document_captions:
        schema_data["document_illustrations"] = document_captions
    if extracted_images:
        schema_data["extracted_images"] = extracted_images
        
    with open(approved_path, "w", encoding="utf-8") as f:
        yaml.dump(schema_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
    # Write edit log to edit_log.json
    # Format of edit_log must be: {"added": [...], "removed": [...], "renamed": {...}}
    formatted_log = {
        "added": edit_log.get("added", []),
        "removed": edit_log.get("removed", []),
        "renamed": edit_log.get("renamed", {})
    }
    with open(edit_log_path, "w", encoding="utf-8") as f:
        json.dump(formatted_log, f, indent=2, ensure_ascii=False)

def merge_details(old_details: Dict[str, Any], new_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Intelligently merges old_details (possibly containing user edits) with newly extracted new_details.
    Does not overwrite user-customized items and combines list values uniquely.
    """
    if not old_details:
        return new_details
        
    merged = {}
    # Retain summary if present in old, otherwise use new
    merged["summary"] = old_details.get("summary") or new_details.get("summary")
    
    # Merge lists keeping order, deduplicate
    for key in ["examples", "common_attributes", "observed_terms", "sample_mentions", "numerical_attributes"]:
        old_list = old_details.get(key, [])
        new_list = new_details.get(key, [])
        
        # Combine lists preserving order, unique items
        combined = list(old_list)
        for item in new_list:
            if item not in combined:
                combined.append(item)
        merged[key] = combined
        
    return merged
