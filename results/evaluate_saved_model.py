import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from model import MaturityCNN, count_parameters
from dataset import create_dataloaders


def evaluate_model(model_path, train_loader, val_loader, device, output_dir='./results'):
    print(f"\nLoading model: {model_path}")

    model = MaturityCNN(num_classes=3, dropout_rate=0.5)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    params = count_parameters(model)

    print("Evaluating training set...")
    train_correct = train_total = 0
    train_preds, train_labels = [], []

    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

    train_acc = 100. * train_correct / train_total

    print("Evaluating validation set...")
    val_correct = val_total = 0
    val_preds, val_labels = [], []
    inference_times = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            start = time.time()
            outputs = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start
            inference_times.append(elapsed / images.size(0) * 1000)

            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = 100. * val_correct / val_total
    avg_inference_time = np.mean(inference_times)
    f1_macro = f1_score(val_labels, val_preds, average='macro')
    f1_weighted = f1_score(val_labels, val_preds, average='weighted')

    return {
        'train_acc': train_acc, 'val_acc': val_acc,
        'f1_macro': f1_macro, 'f1_weighted': f1_weighted,
        'params': params, 'inference_time': avg_inference_time,
        'val_preds': np.array(val_preds), 'val_labels': np.array(val_labels),
        'train_preds': np.array(train_preds), 'train_labels': np.array(train_labels),
        'best_val_acc': checkpoint.get('best_val_acc', 0)
    }


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, annot_kws={'size': 14})
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved: {save_path}")


def plot_classification_report(y_true, y_pred, class_names, save_path):
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics = ['precision', 'recall', 'f1-score']
    data = [[report[cls][m] for m in metrics] for cls in class_names]

    plt.figure(figsize=(10, 4))
    sns.heatmap(data, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=metrics, yticklabels=class_names, annot_kws={'size': 12})
    plt.title('Classification Report', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Classification report saved: {save_path}")
    return report


def plot_metrics_summary(results, save_path):
    class_names = ['未熟', '半熟', '熟透']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Accuracy comparison
    x = ['Train Acc', 'Val Acc']
    y = [results['train_acc'], results['val_acc']]
    bars = axes[0, 0].bar(x, y, color=['#3498db', '#2ecc71'])
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].set_title('Accuracy', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy (%)')
    for bar, val in zip(bars, y):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.2f}%', ha='center', fontsize=11)

    # F1 scores
    x = ['F1 (Macro)', 'F1 (Weighted)']
    y = [results['f1_macro'], results['f1_weighted']]
    bars = axes[0, 1].bar(x, y, color=['#e74c3c', '#9b59b6'])
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_title('F1 Score', fontsize=12)
    for bar, val in zip(bars, y):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.4f}', ha='center', fontsize=11)

    # Parameters
    params_m = results['params'] / 1e6
    axes[0, 2].bar(['Parameters'], [params_m], color='#f39c12')
    axes[0, 2].set_title('Parameters', fontsize=12)
    axes[0, 2].set_ylabel('Parameters (M)')
    axes[0, 2].text(0, params_m + 0.1, f'{params_m:.2f}M', ha='center', fontsize=11)

    # Inference time
    axes[1, 0].bar(['Inference Time'], [results['inference_time']], color='#1abc9c')
    axes[1, 0].set_title('Inference Time', fontsize=12)
    axes[1, 0].set_ylabel('Time (ms/image)')
    axes[1, 0].text(0, results['inference_time'] + 0.5,
                    f'{results["inference_time"]:.2f} ms', ha='center', fontsize=11)

    # Per-class accuracy
    per_class_acc = []
    for i, cls in enumerate(class_names):
        mask = np.array(results['val_labels']) == i
        if mask.sum() > 0:
            acc = (np.array(results['val_preds'])[mask] == i).sum() / mask.sum()
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0)

    bars = axes[1, 1].bar(class_names, per_class_acc, color=['#e67e22', '#27ae60', '#8e44ad'])
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Per-Class Accuracy', fontsize=12)
    for bar, val in zip(bars, per_class_acc):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.2%}', ha='center', fontsize=11)

    # Prediction distribution
    unique, counts = np.unique(results['val_preds'], return_counts=True)
    axes[1, 2].pie(counts, labels=[class_names[i] for i in unique], autopct='%1.1f%%',
                   colors=['#e67e22', '#27ae60', '#8e44ad'])
    axes[1, 2].set_title('Prediction Distribution', fontsize=12)

    plt.suptitle('Model Evaluation Summary', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Summary saved: {save_path}")


def plot_sample_predictions(val_loader, model, device, class_names, num_samples=16, save_path=None):
    model.eval()
    indices = np.random.choice(len(val_loader.dataset), min(num_samples, len(val_loader.dataset)), replace=False)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for idx, ax in zip(indices, axes):
        img, true_label = val_loader.dataset[idx]

        with torch.no_grad():
            img_batch = img.unsqueeze(0).to(device)
            output = model(img_batch)
            pred_label = output.argmax(1).item()

        img = img.cpu().permute(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {class_names[true_label]}\nPred: {class_names[pred_label]}',
                     color=color, fontsize=10)
        ax.axis('off')

    plt.suptitle('Sample Predictions', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Sample predictions saved: {save_path}")
    plt.show()


def main():
    from config import config

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nLoading dataset...")
    train_loader, val_loader = create_dataloaders(config.TRAIN_DIR, config.VAL_DIR, batch_size=32)
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")

    model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        return

    results = evaluate_model(model_path, train_loader, val_loader, device, config.OUTPUT_DIR)
    class_names = ['未熟', '半熟', '熟透']

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Train Accuracy: {results['train_acc']:.2f}%")
    print(f"Val Accuracy: {results['val_acc']:.2f}%")
    print(f"Best Val Accuracy (saved): {results['best_val_acc']:.2f}%")
    print(f"F1 (Macro): {results['f1_macro']:.4f}")
    print(f"F1 (Weighted): {results['f1_weighted']:.4f}")
    print(f"Parameters: {results['params']:,} ({results['params']/1e6:.2f}M)")
    print(f"Inference Time: {results['inference_time']:.2f} ms/image")

    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(results['val_labels'], results['val_preds'], target_names=class_names))

    print("\nGenerating visualizations...")
    plot_confusion_matrix(results['val_labels'], results['val_preds'], class_names,
                          os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png'))
    plot_classification_report(results['val_labels'], results['val_preds'], class_names,
                               os.path.join(config.OUTPUT_DIR, 'classification_report.png'))
    plot_metrics_summary(results, os.path.join(config.OUTPUT_DIR, 'evaluation_summary.png'))

    model = MaturityCNN(num_classes=3, dropout_rate=0.5)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    plot_sample_predictions(val_loader, model, device, class_names, num_samples=16,
                            save_path=os.path.join(config.OUTPUT_DIR, 'sample_predictions.png'))

    report_path = os.path.join(config.OUTPUT_DIR, 'evaluation_report_detailed.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Model Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Train Accuracy: {results['train_acc']:.2f}%\n")
        f.write(f"Val Accuracy: {results['val_acc']:.2f}%\n")
        f.write(f"Best Val Accuracy: {results['best_val_acc']:.2f}%\n")
        f.write(f"F1 (Macro): {results['f1_macro']:.4f}\n")
        f.write(f"F1 (Weighted): {results['f1_weighted']:.4f}\n")
        f.write(f"Parameters: {results['params']:,}\n")
        f.write(f"Inference Time: {results['inference_time']:.2f} ms/image\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(results['val_labels'], results['val_preds'], target_names=class_names))
    print(f"Report saved: {report_path}")

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
