import torch
import torch.nn as nn
import torch.nn.functional as F  # 导入F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# 定义CNN模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(3, 6, 5)  # 输入通道数，输出通道数，卷积核大小
        self.pool = nn.MaxPool2d(2, 2)  # 池化层，大小，步长
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 定义全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 输入特征数，输出特征数
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 假设有10个类别

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # 将数据展平，为全连接层准备
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# 准备数据和数据加载器
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 训练模型
for epoch in range(2):  # 多次循环遍历数据集
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # 每2000个mini-batches打印一次
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')