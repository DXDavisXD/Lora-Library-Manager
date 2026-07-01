"""Standalone python server launcher for LoRA Library Manager."""

from pathlib import Path
from main import setup_project
from server import start_server

if __name__ == "__main__":
    lora_root = setup_project()
    start_server(lora_root, port=8000)
