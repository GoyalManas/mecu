# dependency.py

import subprocess

def install(package):
    """Utility function to install a package using pip."""
    subprocess.check_call(["pip", "install", package])

# Install general dependencies
general_dependencies = [
    "pykeen",
    "networkx",
    "pandas",
    "numpy",
    "scikit-learn",
    "spacy",
    "sentence_transformers"
]

for package in general_dependencies:
    install(package)

# Download Spacy language model
subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_sm"])

# Install PyTorch and PyTorch Geometric dependencies
torch_dependencies = [
    "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
    "torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html",
    "torch-geometric"
]

for package in torch_dependencies:
    install(package)

# Additional dependencies
additional_dependencies = [
    "SPARQLWrapper",
    "networkx"  # (Included again only if used in a different context)
]

for package in additional_dependencies:
    install(package)

print("All dependencies have been successfully installed.")

