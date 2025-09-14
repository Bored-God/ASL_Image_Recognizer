import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

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

test_data = ImageFolder(root = r"E:\unknown\py scripts\datasets\database for asl project (cuda)\asl_alphabet_test", transform=transform)
testloader  = DataLoader(test_data, batch_size=128,shuffle=True)

model = ASLCNN(num_classes=29)
model.load_state_dict(torch.load('/workspaces/ASL_Image_Recognizer/asl_model_size_64x64.pth'))
model.to(device)
model.eval()  # switch to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for images,labels in testloader:
        images,labels = images.to(device),labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs,1)
        total+= labels.size(0)
        correct = (predicted==labels).sum().item()

print(f'Test Accuracy : {100*correct/total:.2f}%')
