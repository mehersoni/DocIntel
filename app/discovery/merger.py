import yaml
from pathlib import Path
from typing import List, Dict, Any
from app.config import CANDIDATE_SCHEMA_PATH

def merge_discovered_categories(
    statistical_cats: List[Dict[str, Any]],
    llm_cats: List[Dict[str, Any]],
    output_path: Path = CANDIDATE_SCHEMA_PATH
) -> List[Dict[str, Any]]:
    """
    Merges category lists from Statistical and LLM discovery.
    Assigns confidence (HIGH if in both, MEDIUM if in one) and tracks sources.
    Saves the output as a YAML file at the specified output_path.
    """
    merged_map = {}
    
    # Process Statistical Categories
    for cat in statistical_cats:
        name = cat["name"].strip()
        norm_name = name.lower()
        merged_map[norm_name] = {
            "name": name,
            "description": cat.get("description", "Discovered via statistical noun phrase frequency analysis and glossary matching."),
            "sources": ["statistical"],
            "confidence": "MEDIUM",
            "match_count": cat.get("match_count", 0)
        }
        
    # Process LLM Categories
    for cat in llm_cats:
        name = cat["name"].strip()
        norm_name = name.lower()
        
        if norm_name in merged_map:
            # Found in both
            merged_item = merged_map[norm_name]
            merged_item["confidence"] = "HIGH"
            if "direct_scan" not in merged_item["sources"]:
                merged_item["sources"].append("direct_scan")
            merged_item["match_count"] = merged_item.get("match_count", 0) + cat.get("match_count", 0)
            # If LLM has a description, use it as it might be richer
            if cat.get("description"):
                merged_item["description"] = cat["description"]
        else:
            # Only in LLM
            merged_map[norm_name] = {
                "name": name,
                "description": cat.get("description", "Discovered directly via direct glossary scanning."),
                "sources": ["direct_scan"],
                "confidence": "MEDIUM",
                "match_count": cat.get("match_count", 0)
            }
            
    # Convert map to list
    merged_list = list(merged_map.values())
    
    # Sort categories: HIGH confidence first, then alphabetical by name
    merged_list.sort(key=lambda x: (x["confidence"] != "HIGH", x["name"]))
    
    # Save as YAML
    schema_data = {"categories": merged_list}
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(schema_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
    return merged_list
