from app.ingestion.extractor import clean_text
from app.ingestion.chunker import chunk_text

def test_clean_text():
    raw_text = "  Hello   World! \n\n\n  This is a   test.  \n"
    expected = "Hello World!\n\nThis is a test."
    assert clean_text(raw_text) == expected

def test_chunk_text():
    text = "word " * 1200
    chunks = chunk_text(text, doc_id="test_doc", chunk_size=500)
    
    assert len(chunks) == 3
    assert chunks[0]["doc_id"] == "test_doc"
    assert chunks[0]["chunk_id"] == "test_doc_chunk_1"
    assert len(chunks[0]["text"].split()) == 500
    assert len(chunks[1]["text"].split()) == 500
    assert len(chunks[2]["text"].split()) == 200
