import pandas as pd

import pandas as pd

def getData(size = None):

    df = pd.read_csv("../data.csv")
    print("total size:", df.size)

    df["Label"] = df["Label"].astype(int)
    if size is not None:
        df = df.head(size)

    texts = df["CommentText"].astype(str).tolist()
    labels = df["Label"].tolist()

    print("Data loading success. Total samples:", len(texts))
    return texts, labels


