import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_REPORTS_DIR = DATA_DIR / "sample_reports"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Create directories if they do not exist
os.makedirs(SAMPLE_REPORTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Settings
DEFAULT_TOP_N = 20

# Artifact file paths
CHUNKS_PATH = OUTPUTS_DIR / "chunks.json"
SCHEMA_DISCOVERY_INPUT_PATH = OUTPUTS_DIR / "schema_discovery_input.json"
CANDIDATE_SCHEMA_PATH = OUTPUTS_DIR / "candidate_schema.yaml"
APPROVED_SCHEMA_PATH = OUTPUTS_DIR / "approved_schema.yaml"
EDIT_LOG_PATH = OUTPUTS_DIR / "edit_log.json"
