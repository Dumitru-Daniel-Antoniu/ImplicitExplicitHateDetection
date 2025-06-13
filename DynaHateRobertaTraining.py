import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from multiprocessing import freeze_support
import os

def main():
    # Load the dataset
    df = pd.read_csv("Balanced_DynaHate_Sample.csv")

    # Filter training and test data
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()

    # Encode labels to integers for binary classification
    le = LabelEncoder()
    train_df['label'] = le.fit_transform(train_df['label'])
    test_df['label'] = le.transform(test_df['label'])

    # Load RoBERTa tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

    # Tokenize the text
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

    # Convert DataFrame to HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

    # Apply tokenization
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # Set format for PyTorch usage
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Load the RoBERTa classification model
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./roberta_output",
        num_train_epochs=2,
        warmup_ratio=0.1,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="no",
        disable_tqdm=False,
        dataloader_num_workers=2,
        fp16=torch.cuda.is_available()
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer for future use
    model_dir = "./roberta_model"
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print(f"Training complete. Model saved")


if __name__ == "__main__":
    freeze_support()
    main()