import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from dataset import MaturityDataset, get_transforms


class ModelEvaluator:
    def __init__(self, model, test_dir, device=None, classes=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = classes or ['未熟', '半熟', '熟透']

        test_dataset = MaturityDataset(test_dir, transform=get_transforms('val'))
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.all_preds = []
        self.all_labels = []
        self.all_probs = []

    def predict(self):
        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Predicting'):
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                self.all_preds.extend(preds.cpu().numpy())
                self.all_labels.extend(labels.numpy())
                self.all_probs.extend(probs.cpu().numpy())

        self.all_preds = np.array(self.all_preds)
        self.all_labels = np.array(self.all_labels)
        self.all_probs = np.array(self.all_probs)

    def calculate_metrics(self):
        metrics = {'accuracy': accuracy_score(self.all_labels, self.all_preds)}

        precision, recall, f1, _ = precision_recall_fscore_support(
            self.all_labels, self.all_preds, average='weighted'
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1
        metrics['per_class'] = classification_report(
            self.all_labels, self.all_preds, target_names=self.classes, output_dict=True
        )
        return metrics

    def plot_confusion_matrix(self, save_path=None):
        cm = confusion_matrix(self.all_labels, self.all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return cm

    def plot_roc_curves(self, save_path=None):
        n_classes = len(self.classes)
        y_true_bin = label_binarize(self.all_labels, classes=range(n_classes))

        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(self.classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], self.all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def generate_report(self, save_path=None):
        self.predict()
        metrics = self.calculate_metrics()

        report = "=" * 60 + "\n"
        report += "Model Evaluation Report\n"
        report += "=" * 60 + "\n\n"
        report += f"Test samples: {len(self.all_labels)}\n\n"
        report += "-" * 40 + "\nOverall Metrics\n" + "-" * 40 + "\n"
        report += f"Accuracy:  {metrics['accuracy']*100:.2f}%\n"
        report += f"Precision: {metrics['precision']*100:.2f}%\n"
        report += f"Recall:    {metrics['recall']*100:.2f}%\n"
        report += f"F1 Score:  {metrics['f1_score']*100:.2f}%\n\n"
        report += "-" * 40 + "\nPer-Class Metrics\n" + "-" * 40 + "\n"

        for class_name in self.classes:
            m = metrics['per_class'][class_name]
            report += f"\n{class_name}:\n"
            report += f"  Precision: {m['precision']*100:.2f}%\n"
            report += f"  Recall:    {m['recall']*100:.2f}%\n"
            report += f"  F1:        {m['f1-score']*100:.2f}%\n"
            report += f"  Samples:   {int(m['support'])}\n"

        print(report)
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved: {save_path}")
        return metrics


def main():
    from config import config
    from model import get_model

    model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        return

    print("Loading model...")
    model = get_model(model_name='cnn', num_classes=config.NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    eval_dir = config.TEST_DIR if os.path.exists(config.TEST_DIR) and os.listdir(config.TEST_DIR) else config.VAL_DIR
    print(f"Using dataset: {eval_dir}")

    evaluator = ModelEvaluator(model, eval_dir, classes=config.CLASSES)
    evaluator.generate_report(save_path=os.path.join(config.OUTPUT_DIR, 'evaluation_report.txt'))
    evaluator.plot_confusion_matrix(save_path=os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png'))
    evaluator.plot_roc_curves(save_path=os.path.join(config.OUTPUT_DIR, 'roc_curves.png'))


if __name__ == '__main__':
    main()
