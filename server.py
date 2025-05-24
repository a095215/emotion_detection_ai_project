from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import threading

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = BertForSequenceClassification.from_pretrained("dfafdsaf/bert_sentiment").to(device)
tokenizer = BertTokenizer.from_pretrained("dfafdsaf/bert_sentiment")
model.eval()

# 全局變數儲存完整分析結果
latest_full_result = None

def analyze_comments(comments):
    encodings = tokenizer(comments, truncation=True, padding=True, return_tensors="pt")
    encodings = {k: v.to(device) for k, v in encodings.items()}
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    counts = {"Negative": 0, "Neutral": 0, "Positive": 0}
    for p in preds:
        label = label_map[p.item()]
        counts[label] += 1

    total = len(comments)
    result = {k: round(v / total, 2) for k, v in counts.items()}
    return result

@app.route('/analyze', methods=['POST'])
def analyze():
    global latest_full_result

    data = request.json
    print("Received data:", data)

    comments = data.get("comments", [])

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    # 分兩階段處理
    first_batch = comments[:300] if len(comments) > 300 else comments
    result_first = analyze_comments(first_batch)

    # 如果超過 300 則，啟動背景分析
    if len(comments) > 300:
        def background_full_analysis():
            global latest_full_result
            full_result = analyze_comments(comments)
            latest_full_result = full_result
            print("完整分析完成:", full_result)

        threading.Thread(target=background_full_analysis).start()

    return jsonify({
        "partial": True if len(comments) > 300 else False,
        "result": result_first
    })

@app.route('/full_result', methods=['GET'])
def get_full_result():
    global latest_full_result
    if latest_full_result:
        return jsonify({
            "partial": False,
            "result": latest_full_result
        })
    else:
        return jsonify({"status": "Full result not ready yet"}), 202

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
