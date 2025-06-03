from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from get_fine_tuning_data import getData  


model_path = "./deberta_v3_sentiment_100000_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


texts, labels = getData(1000)  
true_labels = labels


batch_size = 32
all_preds = []

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        encoding = tokenizer(batch_texts, truncation=True, padding=True, return_tensors="pt", max_length=512)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().tolist()
        all_preds.extend(preds)


accuracy = accuracy_score(true_labels, all_preds)
f1 = f1_score(true_labels, all_preds, average='macro')

print(f" Accuracy: {accuracy:.4f}")
print(f" F1 Score (Macro): {f1:.4f}")
