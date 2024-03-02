import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from plr_exercise.models.cnn import Net
import wandb
import optuna
from torch.utils.data import Subset
import numpy as np
from functools import partial


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return test_loss, accuracy


def objective(trial, subset_size=None):
    # Setup device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Hyperparameters to be tuned by Optuna
    lr = trial.suggest_float("lr", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024])
    epochs = trial.suggest_int("epochs", 1, 5)
    gamma = trial.suggest_float("gamma", 0.5, 0.99)

    # Data loading code
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)

    if subset_size is not None:
        # Ensure reproducibility
        torch.manual_seed(0)

        # Select a random subset of indices for training and testing
        indices_train = np.random.choice(len(dataset1), subset_size, replace=False)
        indices_test = np.random.choice(len(dataset2), subset_size, replace=False)

        # Create subset datasets
        dataset1 = Subset(dataset1, indices_train)
        dataset2 = Subset(dataset2, indices_test)

    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=False)

    # Model, optimizer, and scheduler setup
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # Training loop
    for epoch in range(epochs):  # Number of epochs can also be a hyperparameter
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()

    # Validation loop
    test_loss, accuracy = test(model, device, test_loader)

    # The objective to be maximized (accuracy) or minimized (loss)
    return test_loss  # or return -test_loss to minimize loss


def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning Script")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of trials for Optuna")
    parser.add_argument(
        "--wandb-project", type=str, default="hyperparameter-tuning", help="Weights & Biases project name"
    )
    parser.add_argument(
        "--subset-size", type=int, default=None, help="Size of the subset to use for training and testing"
    )
    parser.add_argument("--enable-wandb", action="store_true", help="Enables logging to Weights & Biases")

    args = parser.parse_args()

    # Initialize wandb
    if args.enable_wandb:
        wandb.init(
            project=args.wandb_project,
            name="MNIST_CNN_Run_001",  # Custom run name
            config={
                "architecture": "CNN",
                "dataset": "MNIST",
            },
        )
    else:
        # Set wandb to a dummy function or class that does nothing
        class DummyWandB:
            def log(*args, **kwargs):
                pass

        wandb = DummyWandB()

    study = optuna.create_study(direction="minimize")

    def wrapped_objective(trial):
        return objective(trial, subset_size=args.subset_size)

    study.optimize(wrapped_objective, n_trials=args.n_trials)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Optionally, log best trial information to wandb
    wandb.log({"best_accuracy": trial.value})
    for key, value in trial.params.items():
        wandb.log({f"best_{key}": value})


if __name__ == "__main__":
    main()
