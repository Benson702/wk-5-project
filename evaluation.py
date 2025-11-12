import torch
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

# Load test data
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_data = datasets.ImageFolder("processed_data", transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Load model
model = models.resnet18()
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(test_data.classes))
model.load_state_dict(torch.load("emotion_model.pth"))
model.eval()

# Predictions
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

print("Classification Report:\n", classification_report(y_true, y_pred, target_names=test_data.classes))
