import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, classification_report
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration class
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_len = 128
    freeze_n = 0
    batch_size = 16
    lr = 1.0401663679887314e-05
    weight_decay = 0.001
    dropout_rate = 0.1
    warmup_ratio = 0.06
    scheduler_type = 'cosine'
    patience = 3
    max_epochs = 5
    save_plots = True
    plot_dir = "validation_plots_t1_kfold"
    output_model = "t1.pt"

# ComprehensiveEvaluator class containing all plotting and metric report functions
class ComprehensiveEvaluator:
    def __init__(self, save_dir="evaluation_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_comprehensive_training_curves(self, history, save=True):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        epochs = range(1, len(history) + 1)

        val_loss = [h['val_loss'] for h in history]
        train_loss = [h['train_loss'] for h in history]
        val_acc = [h['val_acc'] for h in history]
        val_f1 = [h['val_f1'] for h in history]

        axes[0, 0].plot(epochs, train_loss, 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, val_loss, 'r-', label='Val Loss')
        axes[0, 0].set_title('Loss Curve')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(epochs, val_acc, 'r-o', label='Val Accuracy')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        axes[0, 2].plot(epochs, val_f1, 'b-o', label='Val F1 Score')
        axes[0, 2].set_title('Validation F1 Score')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        if save:
            filename = f"{self.save_dir}/training_curves.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📈 Training curves saved to {filename}")

        plt.tight_layout()
        plt.show()
        return fig

    def plot_confusion_matrix_advanced(self, y_true, y_pred, fold, save=True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS'], 
                   yticklabels=['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS'],
                   cbar_kws={'label': 'Count'})
        ax1.set_title(f'Confusion Matrix (Counts) - Fold {fold}')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Reds', ax=ax2,
                   xticklabels=['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS'], 
                   yticklabels=['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS'],
                   cbar_kws={'label': 'Normalized Count'})
        ax2.set_title(f'Normalized Confusion Matrix - Fold {fold}')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        fig.suptitle(f'Confusion Matrix Analysis - Fold {fold}')
        plt.tight_layout()
        
        if save:
            filename = f"{self.save_dir}/confusion_matrix_fold_{fold}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved: {filename}")
        
        plt.show()
        return fig, cm

    def plot_roc_and_pr_curves(self, y_true, y_probs, fold, save=True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        n_classes = y_probs.shape[1]
        # Binarize the output for multiclass ROC and PR calculation
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

        # Plot ROC for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
            ax1.plot(fpr, tpr, lw=2, label=f'Class {i} ROC (AUC = {roc_auc:.2f})')

        ax1.plot([0, 1], [0, 1], 'k--', lw=2)
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_title(f'ROC Curve - Fold {fold}')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.legend(loc='lower right')
        ax1.grid(True)

        # Plot Precision-Recall for each class
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
            pr_auc = np.trapz(precision, recall)
            ax2.plot(recall, precision, lw=2, label=f'Class {i} PR (AUC = {pr_auc:.2f})')

        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_title(f'Precision-Recall Curve - Fold {fold}')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.legend(loc='lower left')
        ax2.grid(True)

        fig.suptitle(f'ROC and PR Curves - Fold {fold}')
        plt.tight_layout()

        if save:
            filename = f"{self.save_dir}/roc_pr_fold_{fold}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📈 ROC/PR curves saved: {filename}")

        plt.show()
        return fig

    def create_classification_table(self, y_true, y_pred, fold, save=True):
        class_report = classification_report(y_true, y_pred, target_names=['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS'], output_dict=True)
        print(f"\n{'='*80}")
        print(f"CLASSIFICATION REPORT TABLE - Fold {fold}")
        print("="*80)
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        for class_name in ['BACKGROUND', 'OBJECTIVE', 'METHODS', 'RESULTS', 'CONCLUSIONS']:
            precision = class_report[class_name]['precision']
            recall = class_report[class_name]['recall']
            f1 = class_report[class_name]['f1-score']
            support = int(class_report[class_name]['support'])
            print(f"{class_name:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")
        print("-" * 60)
        macro_avg = class_report['macro avg']
        print(f"{'Macro Avg':<15} {macro_avg['precision']:<10.4f} {macro_avg['recall']:<10.4f} {macro_avg['f1-score']:<10.4f} {int(macro_avg['support']):<10}")
        weighted_avg = class_report['weighted avg']
        print(f"{'Weighted Avg':<15} {weighted_avg['precision']:<10.4f} {weighted_avg['recall']:<10.4f} {weighted_avg['f1-score']:<10.4f} {int(weighted_avg['support']):<10}")
        print("-" * 60)

        if save:
            filename = f"{self.save_dir}/classification_report_fold_{fold}.json"
            with open(filename, 'w') as f:
                json.dump(class_report, f, indent=2)
            print(f"\n💾 Classification report saved: {filename}")
        
        return class_report

    def create_comprehensive_evaluation_metrics(self, y_true, y_pred, y_probs, fold, save=True):
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE EVALUATION METRICS - Fold {fold}")
        print("="*80)
        
        acc = accuracy_score(y_true, y_pred)
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        precision_macro, recall_macro, _, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        
        n_classes = y_probs.shape[1]
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        
        roc_auc = 0.0
        for i in range(n_classes):
            try:
                roc_auc += roc_auc_score(y_true_bin[:, i], y_probs[:, i])
            except:
                pass
        roc_auc /= n_classes

        # Confusion matrix is generalized with multilabel
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score (Weighted): {f1_weighted:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print(f"ROC AUC (Macro-average): {roc_auc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        comprehensive_metrics = {
            "accuracy": acc,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist()
        }
        
        if save:
            filename = f"{self.save_dir}/comprehensive_metrics_fold_{fold}.json"
            with open(filename, 'w') as f:
                json.dump(comprehensive_metrics, f, indent=2)
            print(f"💾 Comprehensive metrics saved: {filename}")
        
        return comprehensive_metrics

# Freeze transformer encoder layers if freeze_n > 0
def freeze_transformer_layers(model, freeze_n):
    for i, layer in enumerate(model.roberta.encoder.layer):
        if i < freeze_n:
            for param in layer.parameters():
                param.requires_grad = False

# Training function
def train_and_evaluate():
    print("Loading PubMed RCT dataset...")
    dataset = load_dataset(
        "csv",
        data_files={
            "train": "./data/train.txt",
            "validation": "./data/dev.txt",
            "test": "./data/test.txt"
        },
        delimiter="\t",
        column_names=["label", "text"]
    )
    
    valid_labels = {"BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"}
    dataset = dataset.filter(lambda x: x["label"] in valid_labels)
    
    label_map = {"BACKGROUND": 0, "OBJECTIVE": 1, "METHODS": 2, "RESULTS": 3, "CONCLUSIONS": 4}
    def encode_label(example):
        example["label"] = label_map[example["label"]]
        return example
    dataset = dataset.map(encode_label)
    
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    def preprocess_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=Config.max_len)
    tokenized_train = dataset["train"].map(preprocess_fn, batched=True)
    tokenized_validation = dataset["validation"].map(preprocess_fn, batched=True)
    tokenized_test = dataset["test"].map(preprocess_fn, batched=True)
    
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_validation.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    train_loader = DataLoader(tokenized_train, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(tokenized_validation, batch_size=Config.batch_size)
    test_loader = DataLoader(tokenized_test, batch_size=Config.batch_size)
    
    # Model config with specified dropouts and labels count =5
    config = RobertaConfig.from_pretrained("roberta-base",
                                           hidden_dropout_prob=Config.dropout_rate,
                                           attention_probs_dropout_prob=Config.dropout_rate,
                                           num_labels=5)
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", config=config)
    freeze_transformer_layers(model, Config.freeze_n)
    model = model.to(Config.device)
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.lr,
        weight_decay=Config.weight_decay
    )
    
    total_steps = len(train_loader) * Config.max_epochs
    warmup_steps = int(Config.warmup_ratio * total_steps)
    scheduler_fn = get_cosine_schedule_with_warmup if Config.scheduler_type == 'cosine' else get_linear_schedule_with_warmup
    scheduler = scheduler_fn(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    evaluator = ComprehensiveEvaluator(save_dir=Config.plot_dir)

    best_val_loss = float('inf')
    patience_counter = 0

    history = []

    for epoch in range(1, Config.max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            inputs = {k: v.to(Config.device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(Config.device)

            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            train_preds.extend(preds.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        val_probs = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(Config.device) for k, v in batch.items() if k != 'label'}
                labels = batch['label'].to(Config.device)

                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

        history.append({
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        })

        # Early stopping condition
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.output_model)
            print(f"Model saved at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= Config.patience:
                print(f"Early stopping triggered at epoch {epoch} due to no improvement")
                break

    # Load best model for evaluation on test set
    model.load_state_dict(torch.load(Config.output_model, map_location=Config.device))
    model.eval()

    test_preds = []
    test_labels = []
    test_probs = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(Config.device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(Config.device)
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_probs = np.array(test_probs)

    # Use evaluator for comprehensive evaluation and plotting
    evaluator.plot_comprehensive_training_curves(history)
    evaluator.plot_confusion_matrix_advanced(test_labels, test_preds, fold=0)
    evaluator.plot_roc_and_pr_curves(test_labels, test_probs, fold=0)
    evaluator.create_classification_table(test_labels, test_preds, fold=0)
    evaluator.create_comprehensive_evaluation_metrics(test_labels, test_preds, test_probs, fold=0)

if __name__ == "__main__":
    train_and_evaluate()
