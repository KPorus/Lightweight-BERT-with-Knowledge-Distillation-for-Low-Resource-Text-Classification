import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer, DataCollatorWithPadding
from transformers.optimization import get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Configuration
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len = 128
    teacher_epochs = [2, 3, 4]
    teacher_lrs = [1e-5, 2e-5, 3e-5, 5e-5]  # Added smaller lr for finer tuning
    teacher_batch_sizes = [16, 32]
    teacher_weight_decays = [0.0, 0.01]
    teacher_freeze_configs = [0, 2, 4, 6, 8, 10]  # Number of RoBERTa layers to freeze
    output_teacher_model = "best_teacher_model.pt"

# Verify GPU Availability
print(f"Using device: {Config.device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("No GPU detected. Falling back to CPU.")

# Custom Data Collator
class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.padding_collator = DataCollatorWithPadding(tokenizer)

    def __call__(self, examples):
        features = [{"input_ids": ex["input_ids"], "attention_mask": ex["attention_mask"]} for ex in examples]
        labels = [ex["label"] for ex in examples]
        padded = self.padding_collator(features)
        padded["label"] = torch.tensor(labels, dtype=torch.long)
        return padded

# Freeze Transformer Layers
def freeze_transformer_layers(model, freeze_n):
    """Freeze the specified number of RoBERTa encoder layers."""
    if hasattr(model, 'roberta'):
        backbone = model.roberta
    else:
        raise ValueError("Model does not have 'roberta' attribute.")
    
    # Freeze embeddings if freeze_n > 0
    for param in backbone.embeddings.parameters():
        param.requires_grad = False if freeze_n > 0 else True
    
    # Freeze encoder layers
    for i, layer in enumerate(backbone.encoder.layer):
        for param in layer.parameters():
            param.requires_grad = (i >= freeze_n)
    
    # Ensure classifier is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

# Evaluation Function
def evaluate(model, dataloader, device, model_name="Teacher"):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"],
                yticklabels=["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    # Classification Report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]))
    
    return total_loss / len(dataloader), acc, f1, all_preds, all_labels

# Teacher Training with Plots
def train_teacher(model, train_loader, val_loader, epochs, lr, weight_decay, device, freeze_n):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    best_acc = 0.0
    best_state_dict = None
    train_losses, val_losses, train_accs, val_accs = [], [], [], []  # Fixed unpacking error
    
    print(f"Training teacher with freeze_n={freeze_n} on {device}")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        for batch in tqdm(train_loader, desc=f"Teacher Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"[Teacher] Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device, model_name="Teacher")
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"[Teacher] Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state_dict = model.state_dict()

    # Plot Training and Validation Metrics
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Teacher Loss Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Accuracy')
    plt.plot(range(1, len(val_accs) + 1), val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Teacher Accuracy Curve')
    plt.legend()
    plt.show()
    
    return best_acc, best_state_dict

# Main Function
def main():
    # Load Dataset
    print("Loading PubMed RCT dataset...")
    dataset = load_dataset(
        "csv",
        data_files={
            "train": "data/train.txt",
            "validation": "data/dev.txt",
            "test": "data/test.txt"
        },
        delimiter="\t",
        column_names=["label", "text"]
    )

    # Filter out non-data lines (e.g., document IDs starting with ###)
    valid_labels = {"BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"}
    dataset = dataset.filter(lambda x: x["label"] in valid_labels)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    # Define Label Map
    label_map = {"BACKGROUND": 0, "OBJECTIVE": 1, "METHODS": 2, "RESULTS": 3, "CONCLUSIONS": 4}
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Preprocess Function
    def preprocess(example):
        encoding = tokenizer(
            example["text"],
            max_length=Config.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        label = label_map[example["label"]]
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": label
        }

    # Apply Preprocessing
    train_dataset = train_dataset.map(preprocess)
    val_dataset = val_dataset.map(preprocess)
    test_dataset = test_dataset.map(preprocess)
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Grid Search for Teacher
    print("\nStarting RoBERTa fine-tuning grid search...")
    teacher_results = []
    best_teacher_acc = 0.0
    best_teacher_state = None
    best_teacher_config = None

    for freeze_n in Config.teacher_freeze_configs:
        for epochs in Config.teacher_epochs:
            for lr in Config.teacher_lrs:
                for batch_size in Config.teacher_batch_sizes:
                    for weight_decay in Config.teacher_weight_decays:
                        print(f"\nTesting freeze_n={freeze_n}, epochs={epochs}, lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}")
                        teacher = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5).to(Config.device)
                        freeze_transformer_layers(teacher, freeze_n)
                        data_collator = CustomDataCollator(tokenizer=tokenizer)
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)
                        acc, state_dict = train_teacher(teacher, train_loader, val_loader, epochs, lr, weight_decay, Config.device, freeze_n)
                        teacher_results.append({
                            "freeze_n": freeze_n,
                            "epochs": epochs,
                            "lr": lr,
                            "batch_size": batch_size,
                            "weight_decay": weight_decay,
                            "val_acc": acc
                        })
                        if acc > best_teacher_acc:
                            best_teacher_acc = acc
                            best_teacher_state = state_dict
                            best_teacher_config = {
                                "freeze_n": freeze_n,
                                "epochs": epochs,
                                "lr": lr,
                                "batch_size": batch_size,
                                "weight_decay": weight_decay
                            }
                            torch.save(state_dict, Config.output_teacher_model)
                            print(f"*** New best teacher model saved with val_acc={acc:.4f} ***")

    # Save Grid Search Results
    with open("teacher_grid_results.json", "w") as f:
        json.dump(teacher_results, f, indent=2)

    # Load and Evaluate Best Teacher on Test Set
    print("\nEvaluating best teacher model on test set...")
    teacher = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5).to(Config.device)
    teacher.load_state_dict(torch.load(Config.output_teacher_model))
    test_loader = DataLoader(test_dataset, batch_size=best_teacher_config["batch_size"], collate_fn=data_collator)
    _, teacher_acc, teacher_f1, _, _ = evaluate(teacher, test_loader, Config.device, model_name="Teacher")
    print(f"Teacher Final Test Accuracy: {teacher_acc:.4f}, F1 Score: {teacher_f1:.4f}")

if __name__ == "__main__":
    main()