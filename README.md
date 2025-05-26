1.  If you want to train data with cpu, run:

        python -m pip install torch torchvision torchaudio

    If you want to train with gpu, run:

        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

2.  install other dependency:

        python -m pip install transformers datasets tqdm huggingface_hub flask flask-cors

3.  install dependency for DeBERTa model(option):

        python -m pip install tiktoken protobuf sentencepiece

4.  get api-key:

    a. 在瀏覽器搜尋 google cloud, 點選右上角的控制台
    b. 進到控制台後, 選擇 api 和服務
    c. 在 api 和服務中找到 YouTube Data API v3, 點選憑證, 再按建立憑證-api 金鑰
    d. 把 emotion_extension/content.js 頂端的 API_KEY 金鑰替換為您的憑證金鑰

5.  activate extension

    a. 再瀏覽器 manage extension 裡新增 emtion_extension 並啟動
