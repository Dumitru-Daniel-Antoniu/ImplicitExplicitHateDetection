import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from datasets import Dataset
import numpy as np

def main():
    # Load the test data
    df = pd.read_csv("Balanced_DynaHate_Sample.csv")
    test_df = df[df['split'] == 'test'].copy()

    # Encode labels
    le = LabelEncoder()
    test_df['label'] = le.fit_transform(test_df['label'])

    # Load tokenizer and model from saved directory
    model_dir = "./roberta_model"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
    model = RobertaForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Tokenize test data
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

    test_dataset = test_dataset.map(tokenize, batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Create DataLoader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=64)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Evaluation
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))


if __name__ == "__main__":
    main()
