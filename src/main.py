from torch.utils.data import DataLoader
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

import torch
import network 
import json
import timeit

with open("src/config.json") as file:
    config = json.load(file)

epochs = config["epochs"]
learning_rate = config["learning_rate"]
batch_size = config["batch_size"]

def train_loop(dataloader: DataLoader, model: Module, loss_fn: CrossEntropyLoss, optimizer: Optimizer, device: str):
    size = len(dataloader.dataset)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader: DataLoader, model: Module, loss_fn: CrossEntropyLoss, device: str):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    train_data = FashionMNIST("data", transform=ToTensor(), download=True)
    test_data = FashionMNIST("data", transform=ToTensor(), train=False, download=True)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    # Create dataloaders with pinned memory when CUDA is available
    data_loader_train = DataLoader(train_data, batch_size=batch_size, pin_memory=(device != "cpu"))
    data_loader_test = DataLoader(test_data, batch_size=batch_size, pin_memory=(device != "cpu"))

    # Initialize model, optimizer and loss function
    model = network.Neural_Network().to(device)
    optimizer = torch.optim.SGD(params = model.parameters(), lr = learning_rate)
    loss_fn = CrossEntropyLoss()

    for t in range(epochs):
        print(f"Epoch {t+1}\n-----------------")
        train_loop(
            data_loader_train,
            model,
            loss_fn,
            optimizer,
            device
        )
        test_loop(
            data_loader_test,
            model,
            loss_fn,
            device
        )

if __name__ == "__main__":
    start = timeit.default_timer()
    main()
    end = timeit.default_timer()
    print("Time taken:", end-start)