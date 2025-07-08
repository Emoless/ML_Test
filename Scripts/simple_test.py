from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
import torch

model = DistilBertForTokenClassification.from_pretrained('./furniture_ner_model2')
tokenizer = DistilBertTokenizerFast.from_pretrained('./furniture_ner_model2')
text = "I bought a Windsor Bed and a Glen Mirror from IKEA."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=2)[0]
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
labels = [model.config.id2label[pred.item()] for pred in predictions]
for token, label in zip(tokens, labels):
    print(f"{token}: {label}")