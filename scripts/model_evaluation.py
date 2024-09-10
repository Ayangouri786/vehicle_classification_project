import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn as nn

def evaluate_model(model_path, val_path):
    transform = transforms.Compose([
        transforms.Resize((180, 180)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = models.resnet50(weights='DEFAULT')
    model.fc = nn.Linear(model.fc.in_features, len(val_dataset.classes))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=val_dataset.classes)
    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_dataset.classes, yticklabels=val_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def main():
    model_path = 'models/vehicle_dataset/vehicle_test/best_val_loss.pt'
    val_path = 'processed_data/val'
    evaluate_model(model_path, val_path)

if __name__ == "__main__":
    main()
