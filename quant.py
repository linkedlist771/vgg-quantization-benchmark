import torch
import torchvision.models as models
from torchvision import transforms
from torch.quantization import get_default_qconfig, quantize

# 1. Load and modify the pre-trained model
model = models.vgg16(pretrained=False)
model.classifier[6] = torch.nn.Linear(4096, 10)
model.load_state_dict(torch.load("vgg16_finetuned_cifar10.pth"))

# 2. Set the model to evaluation mode
model.eval()

# 3. Specify the quantization configuration
# In this case, we are using the default configuration
qconfig = get_default_qconfig('fbgemm')

# 4. Prepare the model for static quantization
model.qconfig = qconfig
torch.quantization.prepare(model, inplace=True)

# 5. Calibrate the model with representative data
# You need to run the model on a representative dataset
# Here's an example using random data
input_data = torch.rand(1, 3, 224, 224)
model(input_data)

# 6. Convert the model to a quantized version
torch.quantization.convert(model, inplace=True)

# Now the model is quantized and ready to be used for inference
# save it for later use
torch.save(model.state_dict(), "quantized_vgg16_bits_8.pth")