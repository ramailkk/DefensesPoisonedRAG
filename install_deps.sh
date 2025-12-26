#!/bin/bash
set -e

echo "=== Installing PyTorch 2.0.0 for CUDA 11.7 (A40) ==="
pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 torchaudio==2.0.0+cu117 --index-url https://download.pytorch.org/whl/cu117

echo "=== Installing Core ML/NLP Libraries ==="
pip install transformers==4.35.2
pip install sentence-transformers==2.2.2
pip install huggingface_hub==0.25.0

echo "=== Installing RAG & Retrieval Dependencies ==="
pip install beir
pip install faiss-cpu
pip install scikit-learn

echo "=== Installing Data & Utilities ==="
pip install numpy==1.24.3
pip install loguru
pip install lmdeploy
pip install requests
pip install beautifulsoup4
pip install lxml

echo "=== Installing Evaluation Metrics ==="
pip install rouge_score
pip install nltk
pip install evaluate

echo "=== Installing Optional (APIs) ==="
pip install google-generativeai
pip install openai

echo "=== Verifying Installation ==="
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"
python -c "import sentence_transformers; print(f'✓ Sentence Transformers installed')"
python -c "import numpy as np; print(f'✓ NumPy {np.__version__}')"
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"

echo "=== Installation Complete ==="
