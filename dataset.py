import os
import platform
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class MaturityDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['未熟', '半熟', '熟透']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color='gray')

        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders(train_dir, val_dir, batch_size=32, num_workers=4):
    if platform.system() == 'Windows':
        num_workers = 0

    train_dataset = MaturityDataset(train_dir, transform=get_transforms('train'))
    val_dataset = MaturityDataset(val_dir, transform=get_transforms('val'))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader


def get_class_weights(dataset_dir):
    classes = ['未熟', '半熟', '熟透']
    class_counts = []
    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        else:
            count = 0
        class_counts.append(count)

    total = sum(class_counts)
    weights = [total / (len(classes) * count) if count > 0 else 1.0 for count in class_counts]
    return torch.FloatTensor(weights)


def visualize_samples(dataset_dir, num_samples=5):
    import matplotlib.pyplot as plt

    classes = ['未熟', '半熟', '熟透']
    fig, axes = plt.subplots(len(classes), num_samples, figsize=(15, 9))

    for row, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))][:num_samples]

        for col, img_name in enumerate(images):
            img = Image.open(os.path.join(class_dir, img_name))
            axes[row, col].imshow(img)
            axes[row, col].set_title(class_name if col == 0 else '')
            axes[row, col].axis('off')

    plt.tight_layout()
    return fig
