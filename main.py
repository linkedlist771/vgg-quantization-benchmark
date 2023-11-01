import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.quantization import get_default_qconfig, quantize

import time
test_float = True
test_eight_bit = True
if test_float:
    start_time = time.time()
    # 1. 定义模型结构
    model = models.vgg16()
    model.classifier[6] = torch.nn.Linear(4096, 10)

    # 2. 加载模型权重-> 从微调后的模型中加载
    model.load_state_dict(torch.load("vgg16_finetuned_cifar10.pth"))

    # 3. 设置为评估模式
    model.eval()

    # 4. 进行测试
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    end_time = time.time()
    print(f"Time cost on the float model: {end_time - start_time}  s")
    print(f'Accuracy of the float model on the 10000 CIFAR-10 test images: {100 * correct / total}%')
if test_eight_bit:
    start_time = time.time()

    # 1. Load and modify the pre-trained model
    model = models.vgg16(pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, 10)
    model.load_state_dict(torch.load("vgg16_finetuned_cifar10.pth"))

    # 2. Set the model to evaluation mode
    model.eval()

    # 3. Specify the quantization configuration
    qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # 4. Prepare the model for static quantization
    model.qconfig = qconfig
    torch.quantization.prepare(model, inplace=True)

    # 5. Calibrate the model with representative data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    quant_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)

    # Run the model on some input data
    for input_data, _ in test_loader:
        quant_model(input_data)
        break



    # Save the quantized model
    torch.save(quant_model.state_dict(), "quantized_vgg16_bits_8.pth")

    # 7. Perform inference with the quantized model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # quant_model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, "quant model"):
            # images, labels = images.to(device), labels.to(device)
            # input_data = torch.rand(32, 3, 224, 224)

            outputs = quant_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    print('Time taken:', time.time() - start_time)