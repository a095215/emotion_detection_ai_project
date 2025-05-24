from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn, optim
from get_fine_tuning_data import getData
from tqdm import tqdm

# 檢查是否可用 GPU
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 使用 RoBERTa 的 tokenizer（不是 BertTokenizer）
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# 編碼函式不需更動
def encode(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
    encodings['labels'] = torch.tensor(labels)
    return encodings

# 使用 RoBERTa 的分類模型（num_labels 要設為 3）
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)
model.to(device)

# 載入資料（最多 10000 筆）
texts, labels = getData(10000)

# 編碼、建立資料集
encoded_data = encode(texts, labels)
dataset = TensorDataset(encoded_data["input_ids"], encoded_data["attention_mask"], encoded_data["labels"])
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# 設定 optimizer 和 loss
optimizer = optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# 開始訓練
model.train()
for epoch in range(3):
    print(f"Epoch {epoch + 1}")
    progress_bar = tqdm(loader, desc="Training", leave=True)

    for input_ids, attention_mask, labels in progress_bar:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    progress_bar.set_postfix({"loss": loss.item()})

# 儲存模型與 tokenizer
save_directory = "./roberta_sentiment_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print("Model and tokenizer saved to", save_directory)
