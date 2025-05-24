import pandas as pd

# 讀取原始 CSV
df = pd.read_csv("rawData.csv")  # ← 替換為你的實際檔案名

# 定義情緒對應編碼
label_map = {
    "Negative": 0,
    "Neutral": 1,
    "Positive": 2
}

# 過濾掉不合法的情緒標籤（避免 NaN）
df = df[df["Sentiment"].isin(label_map)]

# 建立數字化的 Label 欄位
df["Label"] = df["Sentiment"].map(label_map)

# 只保留 CommentText 和 Label 欄位
df_out = df[["CommentText", "Label"]]

# 儲存為新 CSV
df_out.to_csv("data.csv", index=False)

print("轉換完成，輸出為 cleaned_comments.csv")
