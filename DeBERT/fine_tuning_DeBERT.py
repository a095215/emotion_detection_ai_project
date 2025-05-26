from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score, f1_score
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from get_fine_tuning_data import getData
from tqdm import tqdm
import matplotlib.pyplot as plt

# ====== 設定參數 ======
model_name = "microsoft/deberta-v3-small"
num_labels = 3
num_epochs = 10
batch_size = 16
lr = 2e-5
patience = 2  # Early stopping 條件
validation_ratio = 0.1
save_path = "./best_deberta_model_10000"

# ====== 裝置檢查 ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ====== 載入模型與 tokenizer ======
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.to(device)

# ====== 載入資料與分割 validation ======
texts, labels = getData(10000)
encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
labels_tensor = torch.tensor(labels)

dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels_tensor)
val_size = int(len(dataset) * validation_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# ====== Optimizer 與 Scheduler ======
optimizer = optim.AdamW(model.parameters(), lr=lr)
total_steps = len(train_loader) * num_epochs
warmup_steps = int(0.1 * total_steps)  # 10% warmup，較保守穩定

scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps= warmup_steps,
    num_training_steps=total_steps
)
#total_steps = len(train_loader) * num_epochs
#scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# ====== 損失函數 ======
#loss_fn = nn.CrossEntropyLoss()

# ====== 記錄每個 epoch 的資訊 ======
epochs_list = []
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []
f1_list = []
lr_list = []

# ====== 訓練 + early stopping ======
best_f1 = 0
patience_counter = 0

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0

    for batch in tqdm(train_loader, desc="Training"):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    avg_train_loss = train_loss / len(train_loader)
    train_acc = correct_train / total_train

    # ====== 驗證階段 ======
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct_val += (preds == labels).sum().item()
            total_val += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct_val / total_val
    val_f1 = f1_score(all_labels, all_preds, average='macro')

    # 紀錄當前 epoch
    epochs_list.append(epoch + 1)
    train_loss_list.append(avg_train_loss)
    val_loss_list.append(avg_val_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    f1_list.append(val_f1)
    lr_list.append(scheduler.get_last_lr()[0])

    print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    print(f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    print(f"F1 (macro): {val_f1:.4f}, LR: {lr_list[-1]:.6f}")

    # ====== early stopping ======
    if val_f1 > best_f1:
        print("\n✅ New best model found. Saving...")
        best_f1 = val_f1
        patience_counter = 0
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("\n⏹️ Early stopping triggered.")
            break

print("\n🏁 Training complete. Best F1 (macro):", best_f1)

# ====== 畫圖與儲存 ======
plot_dir = "analyze_10000_16"
os.makedirs(plot_dir, exist_ok=True)

plt.figure()
plt.plot(epochs_list, train_loss_list, label="Train Loss", marker='o')
plt.plot(epochs_list, val_loss_list, label="Val Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Val Loss")
plt.legend()
plt.grid(True)
plt.savefig("analyze_10000_16/loss_plot.png")

plt.figure()
plt.plot(epochs_list, train_acc_list, label="Train Acc", marker='o')
plt.plot(epochs_list, val_acc_list, label="Val Acc", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Val Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("analyze_10000_16/accuracy_plot.png")

plt.figure()
plt.plot(epochs_list, f1_list, label="F1 Score (macro)", marker='o', color='green')
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("F1 Score (Macro) per Epoch")
plt.grid(True)
plt.savefig("analyze_10000_16/f1_plot.png")

plt.figure()
plt.plot(epochs_list, lr_list, label="Learning Rate", marker='o', color='purple')
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate per Epoch")
plt.grid(True)
plt.savefig("analyze_10000_16/lr_plot.png")