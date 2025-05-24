from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn, optim
from get_fine_tuning_data import getData
from tqdm import tqdm

# 檢查 GPU 狀況
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 使用 DeBERTa-v3 的 tokenizer 與模型
model_name = "microsoft/deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.to(device)

# 編碼函式
def encode(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
    encodings['labels'] = torch.tensor(labels)
    return encodings

# 載入資料
texts, labels = getData(10000)
encoded_data = encode(texts, labels)

# 建立資料集與 dataloader
dataset = TensorDataset(encoded_data["input_ids"], encoded_data["attention_mask"], encoded_data["labels"])
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Optimizer 與 loss
optimizer = optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

# 訓練迴圈
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

# 儲存模型
save_directory = "./deberta_v3_sentiment_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print("Model and tokenizer saved to", save_directory)
