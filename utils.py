import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def load_unquantized_model(model_path):
    model = models.vgg16(pretrained=False)
    model.classifier[6] = torch.nn.Linear(4096, 10)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_quantized_model(model_path):
    unquant_model = load_unquantized_model(model_path)
    quant_model = torch.quantization.quantize_dynamic(unquant_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
    return quant_model

def get_cifar_10_dataloader():
    # 4. 进行测试
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader