import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_data = ImageFolder(root = r'E:\unknown\py scripts\datasets\database for asl project (cuda)\asl_alphabet_train', transform=transform)
test_data = ImageFolder(root = r"E:\unknown\py scripts\datasets\database for asl project (cuda)\asl_alphabet_test", transform=transform)

trainloader = DataLoader(train_data,batch_size=128,shuffle=True,num_workers=4,pin_memory=True)
testloader  = DataLoader(test_data, batch_size=128,shuffle=True)

print("classes: ", train_data.classes)

class ASLCNN(nn.Module):
    def __init__(self, num_classes):
        super(ASLCNN,self).__init__()
        self.conv1 = nn.Conv2d(3 ,32,3,1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.pool = nn.MaxPool2d(2,2)

        self.fc1 = None
        self.fc2 = None
        self._initalize_fc(num_classes)
    
    def _initalize_fc(self,num_classes):
        dummy = torch.zeros(1,3,128,128)
        out = self.pool(F.relu(self.conv1(dummy)))
        out = self.pool(F.relu(self.conv2(out)))
        flattened_size = out.view(-1).shape[0]
        self.fc1 = nn.Linear(flattened_size,128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self,x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = torch.flatten(x,1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x
    
num_classes = len(train_data.classes)
model = ASLCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)\

images, labels = next(iter(trainloader))
print(images.shape)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images,labels in trainloader:
        images,  labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader)}")

model.eval()
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
b = "128x128"
torch.save(model.state_dict(), f"asl_model.pth")