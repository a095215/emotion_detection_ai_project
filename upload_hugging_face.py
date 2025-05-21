from huggingface_hub import login
from transformers import BertTokenizer, BertForSequenceClassification

login("withYourFaceHubToken")

model = BertForSequenceClassification.from_pretrained("./bert_sentiment_model")
tokenizer = BertTokenizer.from_pretrained("./bert_sentiment_model")


model.push_to_hub("bert_sentiment")
tokenizer.push_to_hub("bert_sentiment")