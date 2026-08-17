import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from app.config import SCHEMA_DISCOVERY_INPUT_PATH

def select_representative_chunks(chunks: List[Dict[str, str]], max_chunks: int = 5) -> List[str]:
    """
    Selects up to max_chunks representative text chunks.
    If the number of chunks is <= max_chunks, returns all.
    Otherwise, picks spaced-out chunks (e.g. start, middle, end).
    """
    if len(chunks) <= max_chunks:
        return [c["text"] for c in chunks]
        
    step = len(chunks) / max_chunks
    indices = [int(i * step) for i in range(max_chunks)]
    return [chunks[idx]["text"] for idx in indices]

def run_direct_glossary_scan(
    chunks: List[Dict[str, str]], 
    api_key: str = None
) -> List[Dict[str, Any]]:
    """
    Directly scans representative chunks against the local oilfield glossary
    to identify candidate categories. Saves inputs to schema_discovery_input.json.
    Runs completely offline.
    """
    # Scan ALL chunks in offline mode (no representative chunk limitation needed for parsing)
    all_chunk_texts = [c["text"] for c in chunks]
    
    # Select representative chunks for the UI preview file
    rep_chunk_texts = select_representative_chunks(chunks, max_chunks=5)
    
    # Save to outputs/schema_discovery_input.json
    try:
        discovery_input = {"sample_chunks": rep_chunk_texts}
        with open(SCHEMA_DISCOVERY_INPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(discovery_input, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
        
    # Load glossary
    glossary_path = os.path.join("data", "oilfield_glossary.json")
    if not os.path.exists(glossary_path):
        return []
        
    with open(glossary_path, "r", encoding="utf-8") as f:
        glossary = json.load(f)
        
    detected_categories = {}
    
    # Scan text for glossary terms (case-insensitive)
    combined_text = "\n".join(all_chunk_texts).lower()
    
    for key, entry in glossary.items():
        # Match term key or any synonyms
        matches = [key] + entry.get("synonyms", [])
        matched = False
        for m in matches:
            if re.search(r'\b' + re.escape(m) + r'\b', combined_text):
                matched = True
                break
                
    # Count frequency of matches per category
    category_matches = {}
    category_match_frequencies = {}
    
    for key, entry in glossary.items():
        matches = [key] + entry.get("synonyms", [])
        matched_count = 0
        for m in matches:
            occurrences = len(re.findall(r'\b' + re.escape(m) + r'\b', combined_text))
            if occurrences > 0:
                matched_count += occurrences
                
        if matched_count > 0:
            cat_name = entry["category"]
            formatted_cat_name = "".join([w.capitalize() for w in cat_name.split()])
            
            if formatted_cat_name not in category_matches:
                category_matches[formatted_cat_name] = []
                category_match_frequencies[formatted_cat_name] = 0
                
            term = entry["term"]
            if term not in category_matches[formatted_cat_name]:
                category_matches[formatted_cat_name].append(term)
            category_match_frequencies[formatted_cat_name] += matched_count
            
    result = []
    for cat_name, matched_terms in category_matches.items():
        freq = category_match_frequencies[cat_name]
        # Noise filter: Require at least 2 occurrences or at least 2 distinct terms
        if len(matched_terms) >= 2 or freq >= 2:
            description_map = {
                "Geology": "Geological and stratigraphic concepts, rock layers, basins, and lithology formations.",
                "Geophysics": "Physical measurements, petrophysical logs, resistivity, and density metrics.",
                "GeophysicsWells": "Well logging tools, measurement procedures, caliper logs, and wireline logs.",
                "Sedimentology": "Sediment deposition, compaction processes, sedimentary rock facies, and shale/sandstone reservoirs.",
                "Palynology": "Palynological studies, spore and pollen markers, palynomorph microfossils, and biostratigraphy.",
                "Paleontology": "Preserved fossils, biozone intervals, paleoecology, and prehistoric species classification.",
                "Production": "Hydrocarbon flow extraction operations, artificial lift, flow rates, and separators.",
                "WellCompletion": "Downhole tubulars, production casing, perforations, packer configurations, and christmas tree assemblies.",
                "Drilling": "Hole-boring operations, drill bit rotation, mud motor performance, and BHA casing programs.",
                "WellStimulation": "Productivity treatments, slickwater hydraulic fracturing stages, acidizing, and proppants.",
                "Seismic": "Elastic sound waves, acoustic impedance contrasts, reflectors, and vertical travel times.",
                "SeismicAquisition": "Seismic survey design, geophone/hydrophone receiver spreads, vibroseis sources, and shotpoints."
            }
            desc = description_map.get(
                cat_name, 
                f"Technical category related to {cat_name} operations."
            )
            terms_str = ", ".join(matched_terms[:4])
            full_desc = f"{desc} Characterized by terms like: {terms_str}."
            result.append({
                "name": cat_name,
                "description": full_desc,
                "match_count": freq
            })
            
    return result

def generate_category_summary(
    category_name: str, 
    matched_examples: set, 
    matched_terms: set
) -> str:
    """
    Generates a unique, domain-aware summary sentence for a category
    without repeating boilerplate sentence structures across different categories.
    """
    examples_list = list(matched_examples)[:3]
    terms_list = [t for t in matched_terms if t.capitalize() not in examples_list][:3]
    
    ex_str = ", ".join(examples_list) if examples_list else ""
    terms_str = ", ".join(terms_list) if terms_list else ""
    
    domain_summary_templates = {
        "Geology": {
            "with_both": f"Geological evaluation in the document highlights lithological profiles and structural formations like {ex_str}, incorporating parameters such as {terms_str}.",
            "with_ex": f"Geological concepts featured in the document center on lithological units and formations including {ex_str}.",
            "default": "Encompasses geological rock formations, lithological profiles, structural horizons, and stratigraphic sequences."
        },
        "Geophysics": {
            "with_both": f"Subsurface physical logging details petrophysical responses and formation properties like {ex_str}, tracking measurements such as {terms_str}.",
            "with_ex": f"Petrophysical log evaluations center on physical measurements and subsurface responses including {ex_str}.",
            "default": "Encompasses subsurface physical properties, petrophysical logs, resistivity, and density metrics."
        },
        "GeophysicsWells": {
            "with_both": f"Borehole geophysics logging records document tool responses and measurement profiles such as {ex_str}, analyzing indicators like {terms_str}.",
            "with_ex": f"Borehole logging observations highlight tool measurements and well logs including {ex_str}.",
            "default": "Encompasses well logging tools, measurement parameters, caliper logs, and wireline operations."
        },
        "Sedimentology": {
            "with_both": f"Depositional environment and reservoir facies analyses discuss sedimentary features like {ex_str}, evaluating attributes such as {terms_str}.",
            "with_ex": f"Sedimentological assessment focuses on depositional rock facies and reservoir characteristics including {ex_str}.",
            "default": "Encompasses sedimentary rock layers, depositional environments, compaction, and porosity/permeability facies."
        },
        "Palynology": {
            "with_both": f"Palynological biostratigraphy reports organic-walled pollen and microfossil markers including {ex_str}, recording parameters like {terms_str}.",
            "with_ex": f"Palynological marker discussions highlight microfossil spore and pollen assemblages including {ex_str}.",
            "default": "Encompasses palynological microfossils, organic-walled pollen/spore markers, and biostratigraphy."
        },
        "Paleontology": {
            "with_both": f"Biostratigraphic index markers and fossil preservation data classify intervals such as {ex_str}, noting attributes like {terms_str}.",
            "with_ex": f"Paleontological classification identifies biostratigraphic zones and preserved fossil markers including {ex_str}.",
            "default": "Encompasses preserved fossils, biostratigraphic index markers, and paleoecological conditions."
        },
        "Production": {
            "with_both": f"Hydrocarbon extraction operations document flow rates and artificial lift systems like {ex_str}, monitoring parameters such as {terms_str}.",
            "with_ex": f"Production performance records focus on flow extraction and separator operations including {ex_str}.",
            "default": "Encompasses hydrocarbon flow rates, artificial lift systems, and production separators."
        },
        "WellCompletion": {
            "with_both": f"Downhole completion schematics detail casing strings, packers, and assembly hardware like {ex_str}, noting specifications such as {terms_str}.",
            "with_ex": f"Wellbore completion details highlight casing programs and downhole hardware assemblies including {ex_str}.",
            "default": "Encompasses wellbore casing completions, packers, perforation intervals, and wellhead assemblies."
        },
        "Drilling": {
            "with_both": f"Drilling operations and hole-boring mechanics document BHA components and rig performance like {ex_str}, measuring parameters such as {terms_str}.",
            "with_ex": f"Drilling mechanics records center on bit performance and mud motor operations including {ex_str}.",
            "default": "Encompasses drilling rig mechanics, rotary drill strings, mud motors, and hole-boring operations."
        },
        "WellStimulation": {
            "with_both": f"Reservoir stimulation treatments outline hydraulic fracturing fluid stages and proppants like {ex_str}, monitoring metrics such as {terms_str}.",
            "with_ex": f"Stimulation treatment data details reservoir fracturing stages and acidizing operations including {ex_str}.",
            "default": "Encompasses reservoir stimulation, hydraulic fracturing fluid stages, proppants, and acidizing."
        },
        "Seismic": {
            "with_both": f"Acoustic wave reflection data highlights impedance contrasts and subsurface horizons like {ex_str}, recording traveltimes such as {terms_str}.",
            "with_ex": f"Seismic reflection observations center on subsurface acoustic impedance and wave traveltimes including {ex_str}.",
            "default": "Encompasses subsurface seismic wave reflectors, acoustic impedance contrasts, and traveltimes."
        },
        "SeismicAquisition": {
            "with_both": f"Seismic survey acquisition design documents sensor arrays and energy sources like {ex_str}, tracking operational parameters such as {terms_str}.",
            "with_ex": f"Seismic acquisition layouts highlight geophone/hydrophone receiver spreads and shotpoints including {ex_str}.",
            "default": "Encompasses seismic survey design, geophone/hydrophone sensor arrays, and energy sources."
        }
    }
    
    if category_name in domain_summary_templates:
        tpl = domain_summary_templates[category_name]
        if ex_str and terms_str:
            return tpl["with_both"]
        elif ex_str:
            return tpl["with_ex"]
        else:
            return f"No direct keyword mentions detected for {category_name} in the sampled text. {tpl['default']}"
    
    # Generic fallback with varied patterns for custom categories
    if ex_str and terms_str:
        return f"Covers {category_name} operations, with specific emphasis on {ex_str} and associated parameters like {terms_str}."
    elif ex_str:
        return f"Document references for {category_name} center on key concepts including {ex_str}."
    else:
        return f"No direct mentions of {category_name} terms were identified in the sampled text."

def extract_category_details(
    category_name: str,
    category_description: str,
    chunks: List[Dict[str, str]],
    api_key: str = None
) -> Dict[str, Any]:
    """
    Offline implementation of extract_category_details.
    Scans the corpus using regex to extract examples, attributes, terms,
    and actual sentence mentions for a given category.
    """
    # Load glossary
    glossary_path = os.path.join("data", "oilfield_glossary.json")
    glossary = {}
    if os.path.exists(glossary_path):
        with open(glossary_path, "r", encoding="utf-8") as f:
            glossary = json.load(f)
            
    # Find all glossary items belonging to this category
    norm_cat_name = category_name.lower().strip()
    category_terms = []
    category_attributes = set()
    category_synonyms = {}
    
    for key, entry in glossary.items():
        entry_cat_name = "".join(entry["category"].split()).lower()
        if entry_cat_name == norm_cat_name or entry["category"].lower() == norm_cat_name:
            category_terms.append(entry)
            category_attributes.update(entry.get("attributes", []))
            category_synonyms[key] = entry["term"]
            for syn in entry.get("synonyms", []):
                category_synonyms[syn] = entry["term"]
                
    # Search document chunks for mentions
    # Search document chunks for mentions and numerical attributes
    matched_examples = set()
    matched_terms = set()
    sample_mentions = []
    numerical_attributes = []
    
    num_patterns = [
        r'\b\d{1,5}(?:,\d{3})*(?:\.\d+)?\s*-\s*\d{1,5}(?:,\d{3})*(?:\.\d+)?\s*(?:ft|feet|m|meters|m3/d|bpd|bopd|mmscfd|psi|bar|MPa|ppg|g/cm3|deg\s*F|°C|API)\b',
        r'\b\d{1,5}(?:,\d{3})*(?:\.\d+)?\s*(?:ft|feet|m|meters|m3/d|bpd|bopd|mmscfd|psi|bar|MPa|ppg|g/cm3|deg\s*F|°C|API|inch|in\.|\")\b',
        r'\b\d{1,2}(?:/\d{1,2})?\s*-\s*inch\b'
    ]
    
    # Process text chunks
    for chunk in chunks:
        text = chunk["text"]
        
        # Extract numerical parameters
        for pattern in num_patterns:
            num_matches = re.findall(pattern, text, re.IGNORECASE)
            for m in num_matches:
                clean_m = m.strip()
                if clean_m not in numerical_attributes:
                    numerical_attributes.append(clean_m)

        # Split text into sentences for mention extraction
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            sentence_clean = sentence.strip()
            if not sentence_clean:
                continue
                
            sentence_lower = sentence_clean.lower()
            
            # Check for matches of synonyms or keys
            for syn, official_term in category_synonyms.items():
                if re.search(r'\b' + re.escape(syn) + r'\b', sentence_lower):
                    matched_examples.add(official_term)
                    matched_terms.add(syn)
                    
                    # Store sentence mention (ensure it has context and max length)
                    if len(sentence_clean) > 20 and len(sentence_clean) < 300:
                        formatted_mention = f'"{sentence_clean}" (from {chunk["doc_id"]})'
                        if formatted_mention not in sample_mentions:
                            sample_mentions.append(formatted_mention)
                            
    # Build details dictionary
    examples_list = list(matched_examples)
    # Ensure at least 15-20 keywords are retrieved for examples
    if len(examples_list) < 20:
        for entry in category_terms:
            if len(examples_list) >= 20:
                break
            term_name = entry["term"]
            if term_name not in examples_list:
                examples_list.append(term_name)
                
    attributes_list = list(category_attributes)
    if not attributes_list:
        attributes_list = ["Depth", "Status", "Operator"]
        
    terms_list = list(matched_terms)
    # Ensure at least 15-20 keywords are retrieved for observed terms
    if len(terms_list) < 20:
        for syn in category_synonyms.keys():
            if len(terms_list) >= 20:
                break
            if syn not in terms_list:
                terms_list.append(syn)
        
    if not sample_mentions:
        sample_mentions = ["No specific mentions found in the supplied documents."]
    else:
        sample_mentions = sample_mentions[:4]  # Limit to top 4 mentions
        
    # Build summary dynamically
    summary_text = generate_category_summary(category_name, matched_examples, matched_terms)
    
    return {
        "summary": summary_text,
        "examples": examples_list[:5],
        "common_attributes": attributes_list[:6],
        "observed_terms": terms_list[:6],
        "sample_mentions": sample_mentions,
        "numerical_attributes": numerical_attributes[:6]
    }
