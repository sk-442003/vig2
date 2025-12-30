"""Evaluate face ResNet model on data/face/example and save report."""
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import torch
import os
from sklearn.metrics import classification_report

os.makedirs('models/eval_reports', exist_ok=True)
print('Running face evaluation...')

dataset = ImageFolder('data/face/example', transform=transforms.Compose([
    transforms.Resize((48,48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(dataset.classes))
ck = torch.load('models/face_resnet_example_1000.pth', map_location=device)
model.load_state_dict(ck['model_state_dict'])
model.to(device).eval()
trues=[]; preds=[]
with torch.no_grad():
    for X,y in loader:
        X = X.to(device)
        out = model(X)
        p = torch.argmax(out, dim=1).cpu().numpy().tolist()
        preds.extend(p); trues.extend(y.numpy().tolist())
report = classification_report(trues, preds, target_names=dataset.classes)
print(report)
open('models/eval_reports/face_report.txt','w',encoding='utf-8').write(report)
print('Saved models/eval_reports/face_report.txt')