import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class ASLCNN(nn.Module):
    def __init__(self, num_classes=29):
        super(ASLCNN,self).__init__()
        self.conv1 = nn.Conv2d(3 ,32,3,1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.pool= nn.MaxPool2d(2,2)

        self.fc1 = None
        self.fc2 = None
        self.num_classes = num_classes

    def forward(self,x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = torch.flatten(x,1)

        if self.fc1 is None:
            flattned_size = x.shape[1]
            self.fc1 = nn.Linear(flattned_size,128).to(x.device)
            self.fc2 = nn.Linear(128,self.num_classes).to(x.device)

        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print("using: {device}")

model = ASLCNN(num_classes=29)
model.load_state_dict(torch.load("asl_model.pth"))
model.to(device)
model.eval()  # switch to evaluation mode

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])