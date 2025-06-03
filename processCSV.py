import pandas as pd


df = pd.read_csv("rawData.csv")  


label_map = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}


df = df[df["Sentiment"].isin(label_map)]


df["Label"] = df["Sentiment"].map(label_map)


df_out = df[["CommentText", "Label"]]


df_out.to_csv("data.csv", index=False)

print("轉換完成，輸出為 cleaned_comments.csv")
