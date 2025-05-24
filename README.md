1.  If you want to train data with cpu, run:

        python -m pip install torch torchvision torchaudio

    If you want to train with gpu, run:

        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

2.  install other dependency:

        python -m pip install transformers datasets tqdm huggingface_hub

3.  install dependency for DeBERTa model:

        python -m pip install tiktoken protobuf sentencepiece
