"""
Configuration constants to run analysis on UNF servers
"""

from pathlib import Path

# Default paths
DEFAULT_DATASET_DIR = Path("/data/brambati/dataset/ConnectedSpeech-UCSF")
DEFAULT_TRANSCRIPTS_DIR = DEFAULT_DATASET_DIR / "sourcedata/NEW"
DEFAULT_DERIVATIVES_DIR = DEFAULT_DATASET_DIR / "derivatives"
DEFAULT_SPEECHMETRYFLOW_DIR = DEFAULT_DERIVATIVES_DIR / "speechmetryflow_0.1.0" / "results"
DEFAULT_DATABASE_PATH = DEFAULT_SPEECHMETRYFLOW_DIR / "ucsf_preprocessed_all_features.csv"