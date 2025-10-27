import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer, DataCollatorWithPadding
from transformers.optimization import get_linear_schedule_with_warmup
from datasets import load_dataset
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
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
    teacher_lrs = [2e-5, 3e-5, 5e-5]
    teacher_batch_sizes = [16, 32]
    teacher_weight_decays = [0.0, 0.01]
    student_epochs = 10
    student_lrs = [1e-5, 2e-5, 5e-5]
    student_temperatures = [2.0, 4.0]
    output_model = "best_student_model.pt"
    word2vec_size = 300
    student_embed_dims = [300, 512]
    student_hidden_dims = [256, 512]
    student_num_layers = [4, 6]
    student_num_heads = [4, 8]
    student_dropout = 0.1

# Verify GPU Availability
print(f"Using device: {Config.device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
else:
    print("No GPU detected. Falling back to CPU. Please check PyTorch CUDA installation.")

# Custom Data Collator to Preserve Labels
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

# Improved Student Model
class TransformerStudent(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers, num_heads, dropout, word2vec_model, tokenizer):
        super(TransformerStudent, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if word2vec_model is not None:
            for word, idx in tokenizer.get_vocab().items():
                if word in word2vec_model.wv:
                    self.embedding.weight.data[idx] = torch.tensor(word2vec_model.wv[word])
        self.embed_norm = nn.LayerNorm(embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, Config.max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.embed_norm(x)
        x = x + self.pos_encoder[:, :input_ids.size(1), :]
        if attention_mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=~attention_mask.bool())
        else:
            x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)

# Knowledge Distillation Loss
def kd_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5, beta=0.4, gamma=0.1):
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(student_logits / T, dim=-1),
        nn.functional.softmax(teacher_logits / T, dim=-1)
    ) * (T ** 2)
    hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
    seq_loss = nn.MSELoss()(student_logits, teacher_logits)
    return alpha * soft_loss + beta * hard_loss + gamma * seq_loss

# Evaluation Function with Confusion Matrix and Classification Report
def evaluate(model, dataloader, device, is_teacher=False, model_name="Model"):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            if is_teacher:
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
            else:
                logits = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"],
                yticklabels=["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"])
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    # Classification Report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]))
    
    return total_loss / len(dataloader), acc, f1, all_preds, all_labels

# Training Functions with Loss/Accuracy Tracking
def train_teacher(model, train_loader, val_loader, epochs, lr, weight_decay, device):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    best_acc = 0.0
    best_state_dict = None
    patience = 2
    patience_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
    print(f"Training teacher on {device}")
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
        
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device, is_teacher=True, model_name="Teacher")
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"[Teacher] Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state_dict = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    
    # Plot Overfitting/Underfitting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Teacher Loss Over Epochs')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Accuracy')
    plt.plot(range(1, len(val_accs) + 1), val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Teacher Accuracy Over Epochs')
    plt.legend()
    plt.show()
    
    return best_acc, best_state_dict

def train_kd(teacher, student, train_loader, val_loader, epochs, lr, T, device):
    optimizer = optim.AdamW(student.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    best_acc = 0.0
    best_state_dict = None
    patience = 3
    patience_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    
    print(f"Training student on {device}")
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        all_preds, all_labels = [], []
        for batch in tqdm(train_loader, desc=f"KD Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs = teacher(input_ids, attention_mask=attention_mask)
            student_logits = student(input_ids, attention_mask)
            loss = kd_loss(student_logits, teacher_outputs.logits, labels, T=T, alpha=0.5, beta=0.4, gamma=0.1)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            preds = torch.argmax(student_logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"[KD] Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        
        val_loss, val_acc, val_f1, _, _ = evaluate(student, val_loader, device, model_name="Student")
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"[KD] Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state_dict = student.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
    
    # Plot Overfitting/Underfitting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Student Loss Over Epochs')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accs) + 1), train_accs, label='Train Accuracy')
    plt.plot(range(1, len(val_accs) + 1), val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Student Accuracy Over Epochs')
    plt.legend()
    plt.show()
    
    return best_acc, best_state_dict

# Main Script
def main():
    print(f"Using device: {Config.device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("No GPU detected. Falling back to CPU. Please check PyTorch CUDA installation.")

    # Load and Preprocess Dataset
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

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # Subsample dataset
    train_dataset = dataset["train"].shuffle(seed=42).select(range(min(4000, len(dataset["train"]))))
    val_dataset = dataset["validation"].shuffle(seed=42).select(range(min(500, len(dataset["validation"]))))
    test_dataset = dataset["test"].shuffle(seed=42).select(range(min(500, len(dataset["test"]))))

    # Filter out examples with missing or empty text
    train_dataset = train_dataset.filter(lambda x: x["text"] is not None and x["text"].strip() != "")
    val_dataset = val_dataset.filter(lambda x: x["text"] is not None and x["text"].strip() != "")
    test_dataset = test_dataset.filter(lambda x: x["text"] is not None and x["text"].strip() != "")

    # Define label map
    label_map = {
        "BACKGROUND": 0,
        "OBJECTIVE": 1,
        "METHODS": 2,
        "RESULTS": 3,
        "CONCLUSIONS": 4
    }

    # Updated preprocess function
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

    # Apply preprocessing
    train_dataset = train_dataset.map(preprocess)
    val_dataset = val_dataset.map(preprocess)
    test_dataset = test_dataset.map(preprocess)
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # Train Word2Vec Embeddings
    print("Training Word2Vec embeddings...")
    sentences = [tokenizer.decode(example["input_ids"]).split() for example in train_dataset]
    word2vec_model = Word2Vec(sentences, vector_size=Config.word2vec_size, window=5, min_count=1, workers=4)

    # Grid Search for Teacher
    print("\nStarting RoBERTa fine-tuning grid search...")
    teacher_results = []
    best_teacher_acc = 0.0
    best_teacher_state = None
    best_teacher_config = None

    for epochs in Config.teacher_epochs:
        for lr in Config.teacher_lrs:
            for batch_size in Config.teacher_batch_sizes:
                for weight_decay in Config.teacher_weight_decays:
                    print(f"\nTesting epochs={epochs}, lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}")
                    teacher = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5).to(Config.device)
                    data_collator = CustomDataCollator(tokenizer=tokenizer)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)
                    acc, state_dict = train_teacher(teacher, train_loader, val_loader, epochs, lr, weight_decay, Config.device)
                    teacher_results.append({
                        "epochs": epochs,
                        "lr": lr,
                        "batch_size": batch_size,
                        "weight_decay": weight_decay,
                        "val_acc": acc
                    })
                    if acc > best_teacher_acc:
                        best_teacher_acc = acc
                        best_teacher_state = state_dict
                        best_teacher_config = {"epochs": epochs, "lr": lr, "batch_size": batch_size, "weight_decay": weight_decay}

    # Load Best Teacher
    teacher = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5).to(Config.device)
    teacher.load_state_dict(best_teacher_state)

    # Grid Search for Student
    print("\nStarting KD grid search...")
    student_configs = [
        {"lr": lr, "embed_dim": embed_dim, "hidden_dim": hidden_dim, "num_layers": num_layers, "num_heads": num_heads, "T": T}
        for lr in Config.student_lrs
        for embed_dim in Config.student_embed_dims
        for hidden_dim in Config.student_hidden_dims
        for num_layers in Config.student_num_layers
        for num_heads in Config.student_num_heads
        for T in Config.student_temperatures
    ]
    best_student_acc = 0.0
    best_student_state = None
    best_student_config = None

    for config in student_configs:
        print(f"\nTesting student config: {config}")
        student = TransformerStudent(
            vocab_size=tokenizer.vocab_size,
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=5,
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            dropout=Config.student_dropout,
            word2vec_model=word2vec_model,
            tokenizer=tokenizer
        ).to(Config.device)
        param_count = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print(f"Student parameter count: {param_count:,}")
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
        val_loader = DataLoader(val_dataset, batch_size=16, collate_fn=data_collator)
        acc, state_dict = train_kd(
            teacher, student, train_loader, val_loader, Config.student_epochs, config["lr"], config["T"], Config.device
        )
        if acc > best_student_acc:
            best_student_acc = acc
            best_student_state = state_dict
            best_student_config = config
            torch.save(state_dict, Config.output_model)

    # Final Evaluation
    print("\nEvaluating best models on test set...")
    test_loader = DataLoader(test_dataset, batch_size=best_teacher_config["batch_size"], collate_fn=data_collator)
    student = TransformerStudent(
        vocab_size=tokenizer.vocab_size,
        embed_dim=best_student_config["embed_dim"],
        hidden_dim=best_student_config["hidden_dim"],
        output_dim=5,
        num_layers=best_student_config["num_layers"],
        num_heads=best_student_config["num_heads"],
        dropout=Config.student_dropout,
        word2vec_model=word2vec_model,
        tokenizer=tokenizer
    ).to(Config.device)
    student.load_state_dict(best_student_state)

    # Student Metrics
    with torch.profiler.profile(with_stack=True, profile_memory=True) as prof:
        _, student_acc, student_f1, student_preds, student_labels = evaluate(student, test_loader, Config.device, model_name="Student")
    print(f"\nStudent Model Efficiency Metrics:")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
    print(f"Student Model Parameter Count: {param_count:,}")
    print(f"[KD] Final Test Accuracy: {student_acc:.4f}, F1 Score: {student_f1:.4f}")

    # Teacher Metrics
    _, teacher_acc, teacher_f1, teacher_preds, teacher_labels = evaluate(teacher, test_loader, Config.device, is_teacher=True, model_name="Teacher")
    print(f"Teacher Final Test Accuracy: {teacher_acc:.4f}, F1 Score: {teacher_f1:.4f}")

    # Random Forest Baseline
    print("\nTraining Random Forest baseline...")
    tfidf = TfidfVectorizer(max_features=5000)
    train_texts = [example["text"] for example in train_dataset]
    train_labels = [example["label"] for example in train_dataset]
    test_texts = [example["text"] for example in test_dataset]
    test_labels = [example["label"] for example in test_dataset]
    X_train_tfidf = tfidf.fit_transform(train_texts)
    X_test_tfidf = tfidf.transform(test_texts)
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train_tfidf, train_labels)
    rf_preds = rf_model.predict(X_test_tfidf)
    rf_acc = accuracy_score(test_labels, rf_preds)
    rf_f1 = f1_score(test_labels, rf_preds, average='weighted')
    
    # Random Forest Confusion Matrix and Classification Report
    cm = confusion_matrix(test_labels, rf_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"],
                yticklabels=["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"])
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    print("\nRandom Forest Classification Report:")
    print(classification_report(test_labels, rf_preds, target_names=["BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"]))
    print(f"Random Forest Test Accuracy: {rf_acc:.4f}, F1 Score: {rf_f1:.4f}")

    # Save Results
    results = {
        "teacher": teacher_results,
        "student": {"config": best_student_config, "val_acc": best_student_acc, "test_acc": student_acc, "test_f1": student_f1},
        "random_forest": {"test_acc": rf_acc, "test_f1": rf_f1}
    }
    with open("kd_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot Teacher vs Student vs RF Accuracy
    plt.figure(figsize=(6, 4))
    plt.bar(['Teacher', 'Student', 'Random Forest'], [teacher_acc, student_acc, rf_acc], color=['orange', 'blue', 'green'])
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Model Comparison: Test Accuracy')
    plt.show()

if __name__ == "__main__":
    main()