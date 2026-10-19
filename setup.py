import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read() if f.readable() else "Production LLM Twin System with Full LLMOps"

__version__ = "1.0.0"

REPO_NAME = "mlops-project-4-llm-twin-system"
AUTHOR_USER_NAME = "29OmChavan"
SRC_REPO = "llm_twin"
AUTHOR_EMAIL = "omchavan29@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="Production LLM Twin System with 3-Pipeline FTI Architecture, LLaMA/Mistral from Scratch, Advanced Multi-Agent RAG, and Full LLMOps",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
)
