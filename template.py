import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

project_name = "llm_twin"

list_of_files = [
    ".github/workflows/main.yaml",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/llama_mistral_from_scratch.py",
    f"src/{project_name}/components/feature_pipeline.py",
    f"src/{project_name}/components/training_pipeline.py",
    f"src/{project_name}/components/inference_pipeline.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/exception.py",
    "config/config.yaml",
    "params.yaml",
    "main.py",
    "app.py",
    "gradio_app.py",
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
    "setup.py",
    "tests/test_llm_twin.py",
    "docs/IMPLEMENTATION_GUIDE.md",
    "docs/ARCH_DECISION.md",
    "docs/POSTMORTEM.md",
    "docs/HANDOFF_NOTES.md",
    "CHANGELOG.md",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w", encoding="utf-8") as f:
            pass
        logging.info(f"Created empty file: {filepath}")
