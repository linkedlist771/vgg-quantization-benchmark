import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

# 1. 获取预训练的VGG16模型
model = models.vgg16(pretrained=True)

# 修改最后一个全连接层以适应CIFAR-10的10个类别
model.classifier[6] = nn.Linear(4096, 10)

# 2. 加载CIFAR-10数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG16需要224x224的输入
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. 微调模型
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    model.train()

    # 使用 tqdm 来包装 train_loader，以显示进度条
    for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 打印每个epoch的平均损失
    print(f"[Epoch {epoch + 1}] Average loss: {running_loss / len(train_loader):.3f}")
print("Finished Training")

# 4. 保存微调后的模型
torch.save(model.state_dict(), "vgg16_finetuned_cifar10.pth")
