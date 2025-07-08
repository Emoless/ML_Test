from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, DistilBertConfig
from seqeval.metrics import classification_report
import json
import torch

id2label = {0: "O", 1: "B-PRODUCT", 2: "I-PRODUCT"}
label2id = {"O": 0, "B-PRODUCT": 1, "I-PRODUCT": 2}
# Load and update config
config = DistilBertConfig.from_pretrained('./furniture_ner_model')
config.id2label = id2label
config.label2id = label2id

# Load model and tokenizer
model = DistilBertForTokenClassification.from_pretrained('./furniture_ner_model', config=config)
tokenizer = DistilBertTokenizerFast.from_pretrained('./furniture_ner_model')

# Load test dataset
with open('test.json', 'r') as f:
    test_data = json.load(f)

# Prepare predictions
true_labels = []
pred_labels = []
for item in test_data:
    tokens = item['tokens']
    true = item['labels']
    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)[0]
    word_ids = inputs.word_ids()
    pred = [model.config.id2label[pred.item()] for pred in predictions]
    
    # Align predictions with true labels
    aligned_pred = []
    prev_word_id = None
    for word_id, label in zip(word_ids, pred):
        if word_id is None:
            continue
        if word_id != prev_word_id:
            aligned_pred.append(label)
        prev_word_id = word_id
    
    if len(aligned_pred) == len(true):
        true_labels.append(true)
        pred_labels.append(aligned_pred)
    else:
        print(f"Length mismatch for tokens: {tokens}")

# Print classification report
print(classification_report(true_labels, pred_labels))