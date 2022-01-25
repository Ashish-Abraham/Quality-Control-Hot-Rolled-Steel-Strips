from PIL import Image
import torch
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import pickle
import io

# Model class to define the architecture


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # CNNs for rgb images
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        # Connecting CNN outputs with Fully Connected layers
        self.fc1 = nn.Linear(in_features=12*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=1)

    def forward(self, t):
        t = t

        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # print("Here",t.size())
        t = t.reshape(-1, 12*5*5)
        t = self.fc1(t)
        t = F.relu(t)

        t = self.fc2(t)
        t = F.relu(t)

        t = torch.sigmoid(self.out(t))

        return t


# to avoid gradients update
@torch.no_grad()
def predict_vehicle(model, imgdata):
    with open('model_files/labels.json', 'rb') as lb:
        labels = pickle.load(lb)

    loaded_model = model
    loaded_model.load_state_dict(torch.load("model_files/model.pth"))
    loaded_model.eval()

    image = Image.open(io.BytesIO(imgdata))
    resize = transforms.Compose([transforms.Resize((32, 32))])
    image = ToTensor()(image)
    y_result = model(resize(image).unsqueeze(0))
    result_idx = torch.round(y_result)
    for key, value in labels.items():
        if(value == result_idx):
            vehicle = key
            break

    return vehicle