from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parents[1]

GODOT_PROJECT = BASE_DIR / 'script_runner' / 'project'

PYTHON_PACKAGE = 'script_runner'
GDNATIVE_LIBRARY = 'script_runner.gdnlib'
