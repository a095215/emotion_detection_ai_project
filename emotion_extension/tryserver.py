from flask import Flask, request, jsonify
import os
from flask_cors import CORS


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

    # 回傳假情緒比例
    fake_result = {
        "joy": 0.45,
        "anger": 0.15,
        "sadness": 0.30,
        "neutral": 0.10
    }

    return jsonify(fake_result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
