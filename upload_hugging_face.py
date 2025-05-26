# from huggingface_hub import login
# from transformers import BertTokenizer, BertForSequenceClassification

# login("withYourFaceHubToken")

# model = BertForSequenceClassification.from_pretrained("./bert_sentiment_model")
# tokenizer = BertTokenizer.from_pretrained("./bert_sentiment_model")


# model.push_to_hub("bert_sentiment")
# tokenizer.push_to_hub("bert_sentiment")
from huggingface_hub import login
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, DebertaV2ForSequenceClassification, BertTokenizer, RobertaTokenizer, AutoTokenizer


login("")  

models = [
    ("bert_sentiment_5000", "./BERT/best_bert_model_5000"),
    ("bert_sentiment_10000", "./BERT/best_bert_model_10000"),
    ("bert_sentiment_30000", "./BERT/best_bert_model_30000"),
    ("bert_sentiment_50000", "./BERT/best_bert_model_50000"),
    ("roberta_sentiment_30000", "./RoBERT/best_roberta_model_10000"),
    ("roberta_sentiment_30000", "./RoBERT/best_roberta_model_30000"),
    ("roberta_sentiment_50000", "./RoBERT/best_roberta_model_50000"),
    ("roberta_sentiment_100000", "./RoBERT/best_roberta_model_100000"),
    ("deberta_sentiment_5000", "./DeBERT/best_deberta_model_5000"),
    ("deberta_sentiment_50000", "./DeBERT/best_deberta_model_50000"),
]

for repo_name, model_path in models:
    print(f"\nUploading {repo_name} from {model_path}...")

    if "roberta" in repo_name:
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
    elif "deberta" in repo_name:
        model = DebertaV2ForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)

    model.push_to_hub(f"dfafdsaf/{repo_name}")
    tokenizer.push_to_hub(f"dfafdsaf/{repo_name}")

    print(f"{repo_name} uploaded successfully!")

print("\nAll models uploaded successfully!")
