import sys
import os
from pathlib import Path

# Add src folder to sys.path to resolve internal module imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Redirect stdout and stderr to null when running under pythonw (windowless)
# to prevent print() calls from raising AttributeError: 'NoneType' object has no attribute 'write'
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

from main import setup_project
from server import start_server

if __name__ == "__main__":
    lora_root = setup_project()
    start_server(lora_root, port=8000)
