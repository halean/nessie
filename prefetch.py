from transformers import AutoModelForSequenceClassification, AutoTokenizer

m = "bert-base-uncased"
AutoTokenizer.from_pretrained(m)
AutoModelForSequenceClassification.from_pretrained(m)
print('Prefetched model:', m)