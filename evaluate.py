from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model + tokenizer (already trained)
def getEmotionFromComment(comments):


    model = BertForSequenceClassification.from_pretrained("dfafdsaf/bert_sentiment")
    tokenizer = BertTokenizer.from_pretrained("dfafdsaf/bert_sentiment")
    #model = BertForSequenceClassification.from_pretrained("./bert_sentiment_model")
    #tokenizer = BertTokenizer.from_pretrained("./bert_sentiment_model")
    #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #model = BertForSequenceClassification.from_pretrained("./my_trained_model")
    model.eval()

    # Inference
    texts = ["I really enjoyed this.", "Horrible movie."]  # your test inputs
    encodings = tokenizer(comments, truncation=True, padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        
    label_map = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }
    '''
    label_map = {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    }
    '''
    return preds.tolist()
    rtnList = []
    for t, p in zip(texts, preds):
        rtnList.append(label_map[p.item()])
        #print(f"{t} --> {label_map[p.item()]}")

    return rtnList