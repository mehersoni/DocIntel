import os
import json
import re
import spacy
from collections import Counter
from typing import List, Dict, Any
from app.config import DEFAULT_TOP_N

def extract_noun_phrases(chunks: List[Dict[str, str]]) -> List[str]:
    """
    Extracts raw noun phrases from a list of text chunks, removes stopwords/punctuation,
    and returns a list of cleaned phrases. Operates 100% offline without network calls.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
        phrases = []
        for chunk in chunks:
            doc = nlp(chunk.get("text", ""))
            for np in doc.noun_chunks:
                clean_tokens = [
                    token.text.strip().lower()
                    for token in np
                    if not token.is_stop and not token.is_punct and not token.is_digit and len(token.text.strip()) > 1
                ]
                if clean_tokens:
                    phrase = " ".join(clean_tokens)
                    phrases.append(phrase)
        return phrases
    except OSError:
        # Fallback local offline phrase extractor (100% offline, zero network requests)
        stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what", "which",
            "this", "that", "these", "those", "then", "just", "so", "than", "such", "both",
            "through", "about", "against", "between", "into", "throughout", "during", "before",
            "after", "above", "below", "to", "from", "up", "upon", "down", "in", "out", "on",
            "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
            "very", "can", "will", "should", "now"
        }
        phrases = []
        for chunk in chunks:
            text = chunk.get("text", "")
            words = [re.sub(r'[^\w\-]', '', w.lower()) for w in text.split()]
            words = [w for w in words if w and not w.isdigit()]
            for i in range(len(words)):
                for n in range(1, 4):
                    if i + n <= len(words):
                        gram = words[i:i+n]
                        if not all(w in stopwords for w in gram) and len(" ".join(gram)) > 2:
                            phrases.append(" ".join(gram))
        return phrases

import math

def run_statistical_discovery(
    chunks: List[Dict[str, str]], 
    api_key: str = None, 
    top_n: int = DEFAULT_TOP_N
) -> List[Dict[str, Any]]:
    """
    Extracts noun phrases and matches them against the local oilfield glossary database using
    TF-IDF (Term Frequency - Inverse Document Frequency) term weighting across document chunks.
    Evaluates top TF-IDF weighted phrases and applies noise-filtering thresholds.
    """
    raw_phrases = extract_noun_phrases(chunks)
    if not raw_phrases:
        return []
        
    # Load local glossary
    glossary_path = os.path.join("data", "oilfield_glossary.json")
    if not os.path.exists(glossary_path):
        return []
        
    with open(glossary_path, "r", encoding="utf-8") as f:
        glossary = json.load(f)
        
    # Compute TF-IDF weights across chunks
    total_docs = max(len(chunks), 1)
    doc_freq = {}
    phrase_chunk_counts = {}
    
    for chunk in chunks:
        chunk_phrases = extract_noun_phrases([chunk])
        chunk_counts = Counter(chunk_phrases)
        for phrase, count in chunk_counts.items():
            doc_freq[phrase] = doc_freq.get(phrase, 0) + 1
            if phrase not in phrase_chunk_counts:
                phrase_chunk_counts[phrase] = 0
            phrase_chunk_counts[phrase] += count
            
    # Calculate TF-IDF scores
    tfidf_scores = {}
    for phrase, total_tf in phrase_chunk_counts.items():
        df = doc_freq.get(phrase, 1)
        idf = math.log((1 + total_docs) / (1 + df)) + 1.0
        tfidf_scores[phrase] = total_tf * idf
        
    # Select top phrases by TF-IDF score
    eval_n = max(top_n * 5, 150)
    sorted_phrases = sorted(tfidf_scores.keys(), key=lambda p: tfidf_scores[p], reverse=True)
    top_phrases = sorted_phrases[:eval_n]
    
    category_counts = {}
    category_terms = {}
    category_descriptions = {}
    
    # Generic blacklist - restricted to truly non-technical words to prevent false positives
    generic_blacklist = {
        "well", "wells", "date", "time", "limit", "step", "run", "line", 
        "event", "events", "type", "limit", "area", "size", "weight"
    }
    
    for phrase in top_phrases:
        phrase_clean = phrase.lower().strip()
        matched_entry = None
        
        # Tier 1: Exact match or synonym match
        for key, entry in glossary.items():
            if phrase_clean == key or phrase_clean in entry.get("synonyms", []):
                matched_entry = entry
                break
                
        # Tier 2: Check if any glossary key is a substring of the phrase
        if not matched_entry:
            for key, entry in glossary.items():
                if key in phrase_clean:
                    matched_entry = entry
                    break
                    
        # Tier 3: Check if the phrase is a sub-word of the glossary key or synonyms
        if not matched_entry:
            if phrase_clean not in generic_blacklist:
                for key, entry in glossary.items():
                    pattern = r'\b' + re.escape(phrase_clean) + r'\b'
                    if re.search(pattern, key) or any(re.search(pattern, syn) for syn in entry.get("synonyms", [])):
                        matched_entry = entry
                        break
                        
        if matched_entry:
            cat_name = matched_entry["category"]
            formatted_cat_name = "".join([w.capitalize() for w in cat_name.split()])
            
            # Record match details
            category_counts[formatted_cat_name] = category_counts.get(formatted_cat_name, 0) + 1
            if formatted_cat_name not in category_terms:
                category_terms[formatted_cat_name] = set()
            category_terms[formatted_cat_name].add(matched_entry["term"])
            
            # Map description if not set
            if formatted_cat_name not in category_descriptions:
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
                category_descriptions[formatted_cat_name] = description_map.get(
                    formatted_cat_name, 
                    f"Technical category related to {cat_name} operations."
                )
                
    result = []
    for cat_name, terms in category_terms.items():
        count = category_counts[cat_name]
        # Noise filter: Require at least 2 occurrences or at least 2 distinct terms
        # This allows relevant technical words to match while pruning single-word accidental matches
        if count >= 2 or len(terms) >= 2:
            terms_list = list(terms)
            terms_str = ", ".join(terms_list[:4])
            full_desc = f"{category_descriptions[cat_name]} Characterized by terms like: {terms_str}."
            result.append({
                "name": cat_name,
                "description": full_desc,
                "match_count": count
            })
            
    return result
