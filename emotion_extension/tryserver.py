from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from collections import Counter
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from evaluate import getEmotionFromComment

def count_ratios(arr, n):
    counter = Counter(arr)
    total = len(arr)
    ratios = {i: counter.get(i, 0) / total for i in range(n + 1)}
    return ratios


app = Flask(__name__)
CORS(app)


@app.route('/analyze', methods=['POST'])



def analyze():
    data = request.json
    print("🔍 Received data:", data)  # 加這行看實際送進來的內容

    comments = data.get("comments", [])

    # 將留言寫入 txt
    with open("received_comments.txt", "w", encoding="utf-8") as f:
        for c in comments:
            f.write(c + "\n")

    emotionList = getEmotionFromComment(comments[:30])
    print(emotionList)
    percent = count_ratios(emotionList, 2)
    # 回傳假情緒比例
    label_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }
    print(percent[0], percent[1], percent[2])
    result = {
        "Negative": percent[0],
        "Neutral": percent[1],
        "Positive": percent[2]
    }
    fake_result = {
        "joy": 0.45,
        "anger": 0.15,
        "sadness": 0.30,
        "neutral": 0.10
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
