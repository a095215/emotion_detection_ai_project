1.  If you want to train data with cpu, run:

        python -m pip install torch torchvision torchaudio

    If you want to train with gpu, run:

        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

2.  install other dependency:

        python -m pip install transformers datasets tqdm huggingface_hub

3.  serch google cloud in brouser, and click controler in right corner. Choose API service, and then choose YouTube Data API v3.
    If it is first time: create new CA
    otherwise: just click CA

4.  copy the api key into API_KEY in emotion_extension/content.js

5.  open emotion_extension as extension in web-brouser
