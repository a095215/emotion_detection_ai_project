import pandas as pd

import pandas as pd

def getData(size = None):
    # 讀取處理後的 CSV（只包含 CommentText 和 Label）
    df = pd.read_csv("data.csv")
    print("total size:", df.size)
    # 確保 Label 是整數型別
    df["Label"] = df["Label"].astype(int)
    if size is not None:
        df = df.head(size)
    # 取出 texts 和 labels
    texts = df["CommentText"].astype(str).tolist()
    labels = df["Label"].tolist()

    print("Data loading success. Total samples:", len(texts))
    return texts, labels


