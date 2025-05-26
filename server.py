from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import threading
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, help="指定使用的 model", required=True)
parser.add_argument("-a", "--size", type=str, help="model 訓練的大小", required=True)

args = parser.parse_args()

print("指定使用的 mode 是:", args.model)
print("model 訓練的大小是:" , args.size)

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

match args.model:
    case "bert":
        match args.size:
            case "5000":
                model = BertForSequenceClassification.from_pretrained("dfafdsaf/bert_sentiment_5000").to(device)
                tokenizer = BertTokenizer.from_pretrained("dfafdsaf/bert_sentiment_5000")
                print("bert_sentiment_5000")
            case "10000":
                model = BertForSequenceClassification.from_pretrained("dfafdsaf/bert_sentiment_10000").to(device)
                tokenizer = BertTokenizer.from_pretrained("dfafdsaf/bert_sentiment_10000")
                print("bert_sentiment_10000")
            case "30000":
                model = BertForSequenceClassification.from_pretrained("dfafdsaf/bert_sentiment_30000").to(device)
                tokenizer = BertTokenizer.from_pretrained("dfafdsaf/bert_sentiment_30000")
                print("bert_sentiment_30000")
            case "50000":
                model = BertForSequenceClassification.from_pretrained("dfafdsaf/bert_sentiment_50000").to(device)
                tokenizer = BertTokenizer.from_pretrained("dfafdsaf/bert_sentiment_50000")
                print("bert_sentiment_50000")
            case _:
                print("error")
    case "roberta":
        match args.size:
            case "5000":
                model = BertForSequenceClassification.from_pretrained("dfafdsaf/roberta_sentiment_5000").to(device)
                tokenizer = BertTokenizer.from_pretrained("dfafdsaf/roberta_sentiment_5000")
                print("roberta_sentiment_5000")
            case "10000":
                model = BertForSequenceClassification.from_pretrained("dfafdsaf/roberta_sentiment_10000").to(device)
                tokenizer = BertTokenizer.from_pretrained("dfafdsaf/roberta_sentiment_10000")
                print("roberta_sentiment_10000")
            case "30000":
                model = BertForSequenceClassification.from_pretrained("dfafdsaf/roberta_sentiment_30000").to(device)
                tokenizer = BertTokenizer.from_pretrained("dfafdsaf/roberta_sentiment_30000")
                print("roberta_sentiment_30000")
            case "50000":
                model = BertForSequenceClassification.from_pretrained("dfafdsaf/roberta_sentiment_50000").to(device)
                tokenizer = BertTokenizer.from_pretrained("dfafdsaf/roberta_sentiment_50000")
                print("roberta_sentiment_50000")
            case _:
                print("error")
    case "deberta":
        match args.size:
            case "5000":
                model = BertForSequenceClassification.from_pretrained("dfafdsaf/deberta_sentiment_5000").to(device)
                tokenizer = BertTokenizer.from_pretrained("dfafdsaf/deberta_sentiment_5000")
                print("deberta_sentiment_5000")
            case "10000":
                model = BertForSequenceClassification.from_pretrained("dfafdsaf/deberta_sentiment_10000").to(device)
                tokenizer = BertTokenizer.from_pretrained("dfafdsaf/deberta_sentiment_10000")
                print("deberta_sentiment_10000")
            case "30000":
                model = BertForSequenceClassification.from_pretrained("dfafdsaf/deberta_sentiment_30000").to(device)
                tokenizer = BertTokenizer.from_pretrained("dfafdsaf/deberta_sentiment_30000")
                print("deberta_sentiment_30000")
            case "50000":
                model = BertForSequenceClassification.from_pretrained("dfafdsaf/deberta_sentiment_50000").to(device)
                tokenizer = BertTokenizer.from_pretrained("dfafdsaf/deberta_sentiment_50000")
                print("deberta_sentiment_50000")
            case _:
                print("error")
    case _:
        print("error")


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

    return counts
    total = len(comments)
    result = {k: round(v / total, 2) for k, v in counts.items()}
    return result

@app.route('/analyze', methods=['POST'])
def analyze():
    global result
    global count
    global isEnd

    count = 0
    isEnd = False
    result = {"Negative": 0, "Neutral": 0, "Positive": 0}

    data = request.json
    print("Received data:", data)
    comments = data.get("comments", [])
    if not comments:
        return jsonify({"error": "No comments provided"}), 400


    constant = 20
    counter = 0
    while counter < len(comments):
        analyze_result = analyze_comments(comments[counter:min(counter+constant, len(comments))])
        for k in result:
            result[k] += analyze_result[k]
        counter = min(counter+constant, len(comments))
        count = counter
    isEnd = True

@app.route('/get_result', methods=['GET'])
def get_full_result():
    global result
    global count
    global isEnd

    total = sum(result.values())
    if total == 0:
        proportions = {k: 0.0 for k in result}
    else:
        proportions = {k: round(v / total, 2) for k, v in result.items()}

    return jsonify({
        "isEnd": isEnd,
        "result": proportions,
        "count": count
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
