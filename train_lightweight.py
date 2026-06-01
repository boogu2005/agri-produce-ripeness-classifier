import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import get_model, count_parameters, print_model_summary
from dataset import create_dataloaders, get_class_weights


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        class_weights = get_class_weights(config.TRAIN_DIR).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.optimizer = optim.SGD(
            model.parameters(), lr=config.LEARNING_RATE,
            momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY
        )

        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        self.best_val_acc = 0.0
        self.best_epoch = 0

        print(f"Device: {self.device}")
        print(f"Trainable parameters: {count_parameters(self.model):,}")

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

        return running_loss / len(self.train_loader), 100. * correct / total

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validating'):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return running_loss / len(self.val_loader), 100. * correct / total

    def train(self):
        print("\n" + "=" * 60)
        print("Training Started (LightweightCNN)")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(1, self.config.EPOCHS + 1):
            print(f"\nEpoch {epoch}/{self.config.EPOCHS}")
            print("-" * 40)

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.save_model('best_model.pth')
                print(f"*** Best model saved (Val Acc: {val_acc:.2f}%) ***")

            if epoch % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch}.pth')

        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"Total time: {total_time / 60:.2f} min")
        print(f"Model size: ~{os.path.getsize(os.path.join(self.config.MODEL_DIR, 'best_model.pth')) / 1024 / 1024:.2f} MB")
        print("=" * 60)

        self.save_history()
        return self.history

    def save_model(self, filename):
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        filepath = os.path.join(self.config.MODEL_DIR, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }, filepath)

    def save_history(self):
        os.makedirs(self.config.TRAIN_RESULTS_DIR, exist_ok=True)
        filepath = os.path.join(self.config.TRAIN_RESULTS_DIR, 'training_history.png')
        self.plot_history(filepath)

    def plot_history(self, save_path=None):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.history['train_acc'], label='Train Acc')
        axes[1].plot(self.history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(self.history['lr'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('LR Schedule')
        axes[2].grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return fig


def main():
    from config import config

    config.create_dirs()

    if not os.path.exists(config.TRAIN_DIR):
        print("Error: Training data directory not found!")
        print(f"Expected: {config.TRAIN_DIR}")
        print("\nExpected structure:")
        print("train_dataset/")
        print("  ├── 未熟/")
        print("  ├── 半熟/")
        print("  └── 熟透/")
        print("val_dataset/")
        print("  ├── 未熟/")
        print("  ├── 半熟/")
        print("  └── 熟透/")
        return

    print("Loading data...")
    train_loader, val_loader = create_dataloaders(
        config.TRAIN_DIR, config.VAL_DIR,
        batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    print("\nCreating LightweightCNN model...")
    model = get_model(model_name='lightweight', num_classes=config.NUM_CLASSES)
    print_model_summary(model)

    model_size = sum(p.numel() * 4 for p in model.parameters()) / 1024 / 1024
    print(f"\nEstimated model size: {model_size:.2f} MB (will be ~{model_size * 1.1:.2f} MB saved)")

    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()


if __name__ == '__main__':
    main()
