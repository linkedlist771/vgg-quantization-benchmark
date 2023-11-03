import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from utils import load_unquantized_model, load_quantized_model, get_cifar_10_dataloader
import numpy as np

test_float = True
test_eight_bit = True
unquant_model_path = "vgg16_finetuned_cifar10.pth"
unquant_model = load_unquantized_model(unquant_model_path)
quant_model = load_quantized_model(unquant_model_path)
train_loader, test_loader = get_cifar_10_dataloader()

###
# from torch.utils.data import Subset
# num_samples = 30
# subset_indices = list(range(num_samples))
# subset = Subset(test_loader.dataset, subset_indices)
# subset_loader = torch.utils.data.DataLoader(subset, batch_size=test_loader.batch_size, shuffle=False)
# test_loader = subset_loader
###
device = torch.device("cpu")
# torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _test_model(model, loader, device):
    model = model.to(device)
    model.eval()
    start_time = time.time()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    end_time = time.time()
    accuracy = 100 * correct / total
    print(f"Time cost: {end_time - start_time} s")
    print(f'Accuracy on the 10000 CIFAR-10 test images: {accuracy}%')
    class_accuracy = [100 * class_correct[i] / class_total[i] for i in range(10)]
    return accuracy, class_accuracy

def plot_accuracy(accuracy_float, accuracy_quant, classes):
    barWidth = 0.3
    r1 = np.arange(len(accuracy_float))
    r2 = [x + barWidth for x in r1]

    plt.bar(r1, accuracy_float, color='blue', width=barWidth, edgecolor='grey', label='Float Model')
    plt.bar(r2, accuracy_quant, color='red', width=barWidth, edgecolor='grey', label='8-bit Quantized Model')

    plt.xlabel('Categories', fontweight='bold')
    plt.xticks([r + barWidth / 2 for r in range(len(accuracy_float))], classes)
    plt.ylabel('Accuracy')
    plt.title('Comparison of Model Accuracies on CIFAR-10')
    #
    plt.legend(loc='upper right')
    plt.savefig('accuracy.png')
    plt.show()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if test_float:
    print("Testing Float Model")
    # 39.57482862472534 s on GPU
    # 13 min 10 s on CPU
    _, accuracy_float = _test_model(unquant_model, test_loader, device)

if test_eight_bit:
    print("Testing 8-bit Quantized Model")
    _, accuracy_quant = _test_model(quant_model, test_loader, device)

plot_accuracy(accuracy_float, accuracy_quant, classes)
