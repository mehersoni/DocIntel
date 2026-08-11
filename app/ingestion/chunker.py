import json
from pathlib import Path
from typing import List, Dict

def chunk_text(text: str, doc_id: str, chunk_size: int = 500) -> List[Dict[str, str]]:
    """
    Splits text into chunks of approximately chunk_size words.
    """
    words = text.split()
    if not words:
        return []
        
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text_str = " ".join(chunk_words)
        chunk_id = f"{doc_id}_chunk_{i // chunk_size + 1}"
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "text": chunk_text_str
        })
    return chunks

def save_chunks(chunks: List[Dict[str, str]], output_path: Path) -> None:
    """
    Saves chunks to a JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
