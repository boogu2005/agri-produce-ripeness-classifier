import os
import json
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import get_model
from config import config


class MaturityDetector:
    def __init__(self, model_path, device=None, classes=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = classes or config.CLASSES

        self.model = get_model(model_name='cnn', num_classes=len(self.classes))
        self._load_model(model_path)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.model.eval()

    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        print(f"Model loaded on: {self.device}")

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image, self.transform(image).unsqueeze(0)

    def predict(self, image_tensor):
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)

        return {
            'class': self.classes[predicted_idx.item()],
            'confidence': confidence.item(),
            'all_probabilities': {self.classes[i]: prob.item() for i, prob in enumerate(probs[0])}
        }

    def detect_image(self, image_path, show_result=True):
        image, image_tensor = self.preprocess_image(image_path)
        result = self.predict(image_tensor)
        if show_result:
            self._show_result(image, result)
        return result

    def _show_result(self, image, result):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(image)
        ax.axis('off')

        title = f"Prediction: {result['class']}\nConfidence: {result['confidence']*100:.2f}%"
        ax.set_title(title, fontsize=14, pad=10)

        prob_text = "\n".join(
            f"{cls}: {prob*100:.1f}%"
            for cls, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        )
        fig.text(0.02, 0.02, prob_text, fontsize=10, family='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout()
        plt.show()

    def batch_detect(self, image_dir, extensions=None, save_results=True):
        extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in os.listdir(image_dir)
                       if os.path.splitext(f.lower())[1] in extensions]

        if not image_files:
            print(f"No images found in {image_dir}")
            return []

        print(f"Batch detection: {len(image_files)} images...")
        results = []
        for img_file in tqdm(image_files):
            img_path = os.path.join(image_dir, img_file)
            try:
                result = self.detect_image(img_path, show_result=False)
                result['filename'] = img_file
                results.append(result)
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                results.append({'filename': img_file, 'error': str(e)})

        if save_results:
            self._save_batch_results(results, image_dir)

        class_counts = {}
        for r in results:
            if 'class' in r:
                class_counts[r['class']] = class_counts.get(r['class'], 0) + 1

        print("\nDetection summary:")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count} ({count/len(results)*100:.1f}%)")

        return results

    def _save_batch_results(self, results, save_dir):
        output_file = os.path.join(save_dir, 'detection_results.json')
        serializable = []
        for r in results:
            if 'all_probabilities' in r:
                r_copy = r.copy()
                r_copy['all_probabilities'] = {k: float(v) for k, v in r['all_probabilities'].items()}
                r_copy['confidence'] = float(r['confidence'])
                serializable.append(r_copy)
            else:
                serializable.append(r)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"Results saved: {output_file}")


def main():
    model_path = os.path.join(config.MODEL_DIR, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        print("Run train.py first.")
        return

    detector = MaturityDetector(model_path)
    test_image = "test_images/sample.jpg"
    if os.path.exists(test_image):
        result = detector.detect_image(test_image)
        print(f"\nPrediction: {result['class']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")


if __name__ == '__main__':
    main()
