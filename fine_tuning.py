from transformers import BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn, optim
from transformers import BertForSequenceClassification
from get_fine_tuning_data import getData
from tqdm import tqdm

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode(texts, labels):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
    encodings['labels'] = torch.tensor(labels)
    return encodings


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.to(device) 


texts, labels = getData(10000)

encoded_data = encode(texts, labels)
dataset = TensorDataset(encoded_data["input_ids"], encoded_data["attention_mask"], encoded_data["labels"])
loader = DataLoader(dataset, batch_size=8, shuffle=True)

optimizer = optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

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


save_directory = "./bert_sentiment_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print("Model and tokenizer saved to", save_directory)