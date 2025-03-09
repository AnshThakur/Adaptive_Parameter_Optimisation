import torch
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.data import DataLoader, random_split


def get_mnist_loaders(batch_size=256, val_split=0.2):
    # Only applying ToTensor() transformation
    transform = transforms.ToTensor()

    # Download the MNIST dataset
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Split train into training and validation sets
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_mnist, val_mnist = random_split(full_train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader_mnist = DataLoader(train_mnist, batch_size=batch_size, shuffle=True)
    val_loader_mnist = DataLoader(val_mnist, batch_size=batch_size, shuffle=False)
    test_loader_mnist = DataLoader(test_mnist, batch_size=batch_size, shuffle=False)

    return train_loader_mnist, val_loader_mnist, test_loader_mnist

def get_fashion_mnist_loaders(batch_size=256, val_split=0.2):
    # Only applying ToTensor() transformation
    transform = transforms.ToTensor()

    # Download the FashionMNIST dataset
    full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_fmnist = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Split train into training and validation sets
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_fmnist, val_fmnist = random_split(full_train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader_fmnist = DataLoader(train_fmnist, batch_size=batch_size, shuffle=True)
    val_loader_fmnist = DataLoader(val_fmnist, batch_size=batch_size, shuffle=False)
    test_loader_fmnist = DataLoader(test_fmnist, batch_size=batch_size, shuffle=False)

    return train_loader_fmnist, val_loader_fmnist, test_loader_fmnist

def get_kmnist_loaders(batch_size=256, val_split=0.2):
    # Only applying ToTensor() transformation
    transform = transforms.ToTensor()

    # Download the KMNIST dataset
    full_train_dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    test_kmnist = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

    # Split train into training and validation sets
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_kmnist, val_kmnist = random_split(full_train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader_kmnist = DataLoader(train_kmnist, batch_size=batch_size, shuffle=True)
    val_loader_kmnist = DataLoader(val_kmnist, batch_size=batch_size, shuffle=False)
    test_loader_kmnist = DataLoader(test_kmnist, batch_size=batch_size, shuffle=False)

    return train_loader_kmnist, val_loader_kmnist, test_loader_kmnist

# Combine all loaders into one function
def get_all_loaders(batch_size=64, val_split=0.25):
    train_mnist, val_mnist, test_mnist = get_mnist_loaders(batch_size, val_split)
    train_fmnist, val_fmnist, test_fmnist = get_fashion_mnist_loaders(batch_size, val_split)
    train_kmnist, val_kmnist, test_kmnist = get_kmnist_loaders(batch_size, val_split)

    return ([train_mnist,train_fmnist,train_kmnist],  # Train loaders
            [val_mnist,val_fmnist,val_kmnist],        # Validation loaders
            [test_mnist,test_fmnist,test_kmnist]) 


def evaluate_model_avg(model, data_loaders,task_ids):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_correct = 0
    total_samples = 0

    id=0
    with torch.no_grad():
        for data_loader in data_loaders:
            
            task_correct = 0
            task_samples = 0
            
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, id)
                _, preds = torch.max(outputs, 1)
                task_correct += (preds == labels).sum().item()
                task_samples += labels.size(0)

            total_correct += task_correct
            total_samples += task_samples
            
            print(f"Task {id} Accuracy: {100 * task_correct / task_samples:.2f}%")
            id=id+1

    overall_accuracy = 100 * total_correct / total_samples
    return overall_accuracy


def evaluate_model_task(model, data_loader, criterion, task_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs,task_id)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)  # Accumulate loss (multiply by batch size)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Calculate accuracy
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate average loss and accuracy
    average_loss = total_loss / total_samples
    accuracy = 100 * total_correct / total_samples

    return accuracy, average_loss

def load_and_evaluate(model, test_loaders, save_path="best_global"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    print("Best model loaded for final evaluation.")

    # Evaluate the model on the test set
    test_accuracy = evaluate_model_avg(model, test_loaders, nn.CrossEntropyLoss(), num_tasks=3)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
