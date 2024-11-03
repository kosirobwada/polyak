'''Train Tiny ImageNet with PyTorch.'''
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import math
import os
import zipfile
import requests
import csv
import json
from models.wideresnet import build_wide_resnet
from models.resnet import build_resnet
from sgd import SGD
import tempfile

tempfile.tempdir = './tmp'


class TinyImageNetValDataset(Dataset):
    def __init__(self, annotations_map, img_dir, class_to_idx, transform=None):
        self.annotations_map = annotations_map
        self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.image_filenames = list(annotations_map.keys())  # List of image file names

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert('RGB')

        class_label = self.annotations_map[self.image_filenames[idx]]
        label = self.class_to_idx[class_label]

        if self.transform:
            image = self.transform(image)

        return image, label


# Function to download and extract Tiny ImageNet
def download_and_extract_tiny_imagenet(data_dir='./data'):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_filename = os.path.join(data_dir, "tiny-imagenet-200.zip")
    extracted_dir = os.path.join(data_dir, "tiny-imagenet-200")

    # Create the directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download the data if it doesn't exist
    if not os.path.exists(extracted_dir):
        print("Downloading Tiny ImageNet dataset...")
        response = requests.get(url, stream=True)
        with open(zip_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        # Extract the data using zipfile
        print("Extracting Tiny ImageNet dataset...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        print("Tiny ImageNet dataset is ready.")
    else:
        print("Tiny ImageNet dataset already exists.")


def load_validation_annotations():
    annotations_path = './data/tiny-imagenet-200/val/val_annotations.txt'

    # Use a context manager to open the file
    with open(annotations_path, 'r') as file:
        annotations_map = {}

        # Read each line and map the image filename to its class label
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:  # Check for safety to ensure there are enough elements
                image_filename = parts[0]
                class_label = parts[1]
                annotations_map[image_filename] = class_label

    return annotations_map


# Function to calculate the total number of training steps
def calc_total_steps():
    total = 0
    if lr_method == "scaled_bs":
        for i in range(epochs):
            bs = init_bs * (delta ** (i // incr_interval))
            total += math.ceil(len(trainset) / bs)
    else:
        a = (bs_max - init_bs) / (delta ** ((epochs - incr_interval) / incr_interval) - 1)
        b = init_bs - a
        for i in range(epochs):
            bs = min(math.ceil(a * (delta ** (i // incr_interval)) + b), bs_max)
            total += math.ceil(len(trainset) / bs)
    return total


# Function to save the model checkpoint to a file
def save_checkpoint(state):
    print(f"Saving checkpoint to {checkpoint_path}...")
    torch.save(state, checkpoint_path)


# Function to load the model checkpoint from a file
def load_checkpoint():
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    return checkpoint


def train(epoch, steps):
    # Calculate the new batch size based on the strategy
    global batch_size
    if case in ["incr_bs_decay_lr", "incr_bs_incr_lr", "incr_bs_warmup_lr"]:
        if lr_method == "scaled_bs":
            new_batch_size = int(init_bs * (delta ** (epoch // incr_interval)))
        else:
            a = (bs_max - init_bs) / (delta ** ((epochs - incr_interval) / incr_interval) - 1)
            b = init_bs - a
            new_batch_size = min(math.ceil(a * (delta ** (epoch // incr_interval)) + b), bs_max)
    else:
        new_batch_size = batch_size
    print(f'batch size: {new_batch_size}')

    # Update the trainloader with new batch size if it has changed
    if new_batch_size != batch_size:
        global trainloader
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=new_batch_size, shuffle=True, num_workers=2)
        batch_size = new_batch_size

    net.train()
    train_loss = 0
    correct = 0
    total = 0
    p_norm = 0
    last_lr = lr
    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        steps += 1

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if lr_method in ["diminishing", "linear", "poly"]:
            scheduler.step()

        last_lr = scheduler.get_last_lr()[0]
        lr_batch.append([epoch + 1, steps, last_lr, new_batch_size])

    p_norm = get_full_grad_list(net, trainset, optimizer)
    norm_results.append([epoch + 1, steps, p_norm])

    print(f'learning rate: {last_lr}')

    train_accuracy = 100. * correct / total
    train_results.append([epoch + 1, steps, train_loss / (batch_idx + 1), train_accuracy, last_lr])

    return steps


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    test_accuracy = 100. * correct / total
    test_results.append([epoch + 1, test_loss / (batch_idx + 1), test_accuracy])


def get_full_grad_list(net, train_set, optimizer):
    parameters = [p for p in net.parameters()]
    print(f'norm_batch:{batch_size}')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    init = True
    full_grad_list = []

    for i, (xx, yy) in (enumerate(train_loader)):
        xx = xx.to(device, non_blocking=True)
        yy = yy.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss(reduction='mean')(net(xx), yy)
        loss.backward()
        if init:
            for params in parameters:
                full_grad = torch.zeros_like(params.grad.detach().data)
                full_grad_list.append(full_grad)
            init = False

        for i, params in enumerate(parameters):
            g = params.grad.detach().data
            full_grad_list[i] += (batch_size / len(train_set)) * g

    total_norm = 0.0
    for grad in full_grad_list:
        total_norm += grad.norm().item() ** 2

    return total_norm ** 0.5


def diminishing_lr_lambda(steps):
    return 1 / math.sqrt(steps + 1)


def linear_lr_lambda(steps):
    return 1 - (steps / total_steps)


def exp_growth_lr_lambda(epoch):
    gamma = (lr_max / lr) ** (incr_interval / (epochs - incr_interval))
    return min(gamma ** (epoch // incr_interval), (lr_max / lr))


def triply_incr_bs_lr_lambda(epoch):
    a = (lr_max - lr) / (1.5 ** ((epochs - incr_interval) / incr_interval) - 1)
    b = lr - a
    return min((1 / lr) * (a * (1.5 ** (epoch // incr_interval)) + b), (lr_max / lr))


def quadruply_incr_bs_lr_lambda(epoch):
    a = (lr_max - lr) / (1.9 ** ((epochs - incr_interval) / incr_interval) - 1)
    b = lr - a
    return min((1 / lr) * (a * (1.9 ** (epoch // incr_interval)) + b), (lr_max / lr))


def warmup_const_lr_lambda(epoch):
    if epoch < warmup_epochs:
        gamma = (lr_max / lr) ** (warmup_interval / (warmup_epochs - warmup_interval))
        return min(gamma ** (epoch // warmup_interval), (lr_max / lr))
    else:
        return (lr_max / lr)


def warmup_cosine_lr_lambda(epoch):
    if epoch < warmup_epochs:
        gamma = (lr_max / lr) ** (warmup_interval / (warmup_epochs - warmup_interval))
        return min(gamma ** (epoch // warmup_interval), (lr_max / lr))
    else:
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
        return (lr_max / lr) * cosine_decay


def scaled_bs_lr_lambda(epoch):
    return lr_mult ** (epoch // incr_interval)


# Command Line Argument
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training with Schedulers')
    parser.add_argument('config_path', type=str, help='path of config file(.json)')
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    model = config["model"]
    case = config["case"]
    lr = config["lr"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    checkpoint_path = config["checkpoint_path"]
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    lr_method = config["lr_method"]
    if case in ["incr_bs_decay_lr", "incr_bs_incr_lr", "incr_bs_warmup_lr"]:
        if not lr_method == "scaled_bs":
            bs_max = config["bs_max"]
        init_bs = batch_size
        delta = config["bs_delta"]
        incr_interval = config["incr_interval"]
    if case in ["incr_bs_incr_lr", "incr_bs_warmup_lr"]:
        if not lr_method == "scaled_bs":
            lr_max = config["lr_max"]
    if case == "incr_bs_warmup_lr":
        warmup_epochs = config["warmup_epochs"]
        warmup_interval = config["warmup_interval"]
    if lr_method == "poly":
        power = config["power"]
    if lr_method == "scaled_bs":
        lr_mult = config["lr_multiplier"]

    # Dataset Preparation
    download_and_extract_tiny_imagenet(data_dir='./data')
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2302, 0.2265, 0.2262)
    transform_train = transforms.Compose([transforms.RandomCrop(64, padding=8),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(15),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])
    trainset = datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    class_to_idx = trainset.class_to_idx

    annotations_map = load_validation_annotations()
    val_image_dir = './data/tiny-imagenet-200/val/images'
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])
    testset = TinyImageNetValDataset(annotations_map=annotations_map, img_dir=val_image_dir, class_to_idx=class_to_idx, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Device Setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model Preparation
    if model == "ResNet18":
        net = build_resnet(model_type="resnet18", dataset_name='TinyImageNet')
        print("model: ResNet18")
    elif model == "WideResNet28_10":
        net = build_wide_resnet(model_type=model, dataset_name='TinyImageNet')
        print("model: WideResNet28_10")
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()

    if case in ["incr_bs_decay_lr", "incr_bs_incr_lr", "incr_bs_warmup_lr"]:
        total_steps = calc_total_steps()
    else:
        total_steps = (math.ceil(len(trainset) / batch_size)) * epochs
    print("total_steps: ", total_steps)

    optimizer = SGD(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    if lr_method == "diminishing":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, diminishing_lr_lambda)

    elif lr_method == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    elif lr_method == "poly":
        scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=power)

    elif lr_method == "linear":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, linear_lr_lambda)

    elif lr_method == "exp_growth":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, exp_growth_lr_lambda)

    elif lr_method == "warmup_const":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_const_lr_lambda)

    elif lr_method == "warmup_cosine":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_lr_lambda)

    elif lr_method == "triply_incr_bs":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, triply_incr_bs_lr_lambda)

    elif lr_method == "quadruply_incr_bs":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, quadruply_incr_bs_lr_lambda)

    elif lr_method == "scaled_bs":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, scaled_bs_lr_lambda)

    print(optimizer)

    # Lists to save results
    train_results = []
    test_results = []
    norm_results = []
    lr_batch = []

    if args.resume:
        checkpoint = load_checkpoint()
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_results = checkpoint.get('train_results', [])
        test_results = checkpoint.get('test_results', [])
        norm_results = checkpoint.get('norm_results', [])
        lr_batch = checkpoint.get('lr_batch', [])
        steps = checkpoint['steps']
    else:
        start_epoch = 0
        steps = 0
        lr_batch.append([1, steps, lr, batch_size])

    for epoch in range(start_epoch, epochs):
        steps = train(epoch, steps)
        test(epoch)

        if lr_method in ["cosine", "exp_growth", "warmup_const", "warmup_cosine", "triply_incr_bs", "quadruply_incr_bs", "scaled_bs"]:
            scheduler.step()

        save_checkpoint({
            'epoch': epoch,
            'steps': steps,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_results': train_results,
            'test_results': test_results,
            'norm_results': norm_results,
            'lr_batch': lr_batch,
        })

        print(f'Epoch: {epoch + 1}, Steps: {steps}, Train Loss: {train_results[epoch][2]:.4f}, Test Acc: {test_results[epoch][2]:.2f}%')

    # Save to CSV file
    filename = f"../result/csv/TinyImageNet/{model}/{case}/{lr_method}"
    if lr_method == "poly":
        filename = f"{filename}_{power}"
    elif lr_method in ["exp_growth", "warmup_const", "warmup_cosine", "triply_incr_bs", "quadruply_incr_bs"]:
        filename = f"{filename}/min{lr}_max{lr_max}"

    if not os.path.exists(filename):
        os.makedirs(filename)

    with open(f"{filename}/train.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Steps', 'Train Loss', 'Train Accuracy', 'Learning Rate'])
        writer.writerows(train_results)

    with open(f"{filename}/test.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Test Loss', 'Test Accuracy'])
        writer.writerows(test_results)

    with open(f"{filename}/norm.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Steps', 'Full Gradient Norm'])
        writer.writerows(norm_results)

    with open(f"{filename}/lr_batch.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Steps', 'Learning Rate', 'Batch Size'])
        writer.writerows(lr_batch)