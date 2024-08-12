# app.py
from flask import Flask, request, jsonify, render_template
import torch
from torch import nn
from torchvision import models
from torchvision import transforms
from torchvision import datasets
from PIL import Image

app = Flask(__name__)

# Load the model
class BirdClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BirdClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

num_classes = 525  # Replace with the actual number of classes
model = BirdClassifier(num_classes)
model.load_state_dict(torch.load('bird_classifier.pth', map_location=torch.device('cpu')))
model.eval()

# Define the data transformations
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dir = 'train'
valid_dir = 'valid'
test_dir = 'test'

image_datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms),
    'valid': datasets.ImageFolder(valid_dir, data_transforms),
    'test': datasets.ImageFolder(test_dir, data_transforms)
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file.stream)
    image = data_transforms(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    
    class_name = image_datasets['train'].classes[preds[0]]
    return jsonify({'class_name': class_name})

if __name__ == '__main__':
    app.run(debug=True)