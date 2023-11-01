import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.quantization import get_default_qconfig, quantize

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



quant = torch.ao.quantization.QuantStub()

dequant = torch.quantization.DeQuantStub()

# Run the model on some input data
for input_data, _ in test_loader:
    output = model(input_data)

    print(f"output shape before quantization: {output.shape}")
    break

quant_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
for input_data, _ in test_loader:
    # set the input data into QuantizedCPU backend
    output = model(input_data)
    print(f"output shape after quantization: {output.shape}")
    break
# Save the quantized model
torch.save(model.state_dict(), "quantized_vgg16_bits_8.pth")

# # 6. Convert the model to a quantized version
# torch.quantization.convert(model, inplace=True)
# # Run the model on some input data
# for input_data, _ in test_loader:
#     # set the input data into QuantizedCPU backend
#     input_data = quant(input_data)
#     output = model(input_data)
#     print(f"output shape after quantization: {output.shape}")
#     break
# # Save the quantized model
