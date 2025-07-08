from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset
import json
import torch
import numpy as np

# Load datasets
try:
    with open('train.json', 'r') as f:
        train_data = json.load(f)
    with open('test.json', 'r') as f:
        test_data = json.load(f)
except json.JSONDecodeError as e:
    print(f"JSON decode error: {e}")
    exit(1)

# Verify data structure
if not isinstance(train_data, list) or not all(isinstance(item, dict) and 'tokens' in item and 'labels' in item for item in train_data):
    print("Error: train.json must be a list of dictionaries with 'tokens' and 'labels' keys")
    exit(1)
if not isinstance(test_data, list) or not all(isinstance(item, dict) and 'tokens' in item and 'labels' in item for item in test_data):
    print("Error: test.json must be a list of dictionaries with 'tokens' and 'labels' keys")
    exit(1)

# Verify tokens and labels alignment
for i, item in enumerate(train_data + test_data):
    if len(item['tokens']) != len(item['labels']):
        print(f"Error: Mismatch in tokens and labels length at example {i+1}")
        exit(1)

# Define label mappings
label_list = ['O', 'B-PRODUCT', 'I-PRODUCT']
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for idx, label in label2id.items()}

# Initialize tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Function to tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        padding='max_length',
        max_length=128,
        return_tensors='np'
    )
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens get -100
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(label2id[label[word_idx]] if label[word_idx].startswith('I-') else -100)
            previous_word_idx = word_idx
        # Pad or truncate labels to match max_length
        label_ids = label_ids[:128] + [-100] * (128 - len(label_ids)) if len(label_ids) < 128 else label_ids[:128]
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Convert to Hugging Face Dataset
try:
    train_dataset = Dataset.from_dict({
        'tokens': [item['tokens'] for item in train_data],
        'labels': [item['labels'] for item in train_data]
    })
    test_dataset = Dataset.from_dict({
        'tokens': [item['tokens'] for item in test_data],
        'labels': [item['labels'] for item in test_data]
    })
except Exception as e:
    print(f"Error creating dataset: {e}")
    exit(1)

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

# Initialize model
model = DistilBertForTokenClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
try:
    trainer.train()
except Exception as e:
    print(f"Training error: {e}")
    exit(1)

# Save the model
model.save_pretrained('./furniture_ner_model2')
tokenizer.save_pretrained('./furniture_ner_model2')
print("Model and tokenizer saved to ./furniture_ner_model2")