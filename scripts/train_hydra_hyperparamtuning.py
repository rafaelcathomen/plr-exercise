import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from plr_exercise.models.cnn import Net
from plr_exercise import PLR_ROOT_DIR
import wandb
import optuna
from torch.utils.data import Subset
import numpy as np
import hydra
from omegaconf import DictConfig  # , OmegaConf
from pathlib import Path
import os


def train(model, device, train_loader, optimizer, epoch):
    """
    Trains the model for one epoch using the given data.

    Args:
        model (nn.Module): The neural network model to be trained.
        device (torch.device): The device (CPU or GPU) to be used for training.
        train_loader (DataLoader): The data loader for the training dataset.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's parameters.
        epoch (int): The current epoch number.

    Returns:
        None
    """

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    """
    Evaluate the performance of a model on the test dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        device (torch.device): The device to run the evaluation on.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.

    Returns:
        tuple: A tuple containing the test loss and accuracy.
    """
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


def objective(trial, config):
    """Optimization objective function for hyperparameter tuning.

    Args:
        trial (optuna.Trial): A trial object that stores the hyperparameters to be optimized.
        config (Config): An object that contains the configuration settings.

    Returns:
        float: The test loss value.
    """
    # Setup device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Hyperparameters to be tuned by Optuna
    lr = trial.suggest_float("lr", *config.lr_range)
    batch_size = trial.suggest_categorical("batch_size", config.batch_sizes)
    epochs = trial.suggest_int("epochs", *config.epochs_range)
    gamma = trial.suggest_float("gamma", *config.gamma_range)

    # Data loading code
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)

    if config.subset_size is not None:
        # Ensure reproducibility
        torch.manual_seed(0)

        # Select a random subset of indices for training and testing
        indices_train = np.random.choice(len(dataset1), config.subset_size, replace=False)
        indices_test = np.random.choice(len(dataset2), config.subset_size, replace=False)

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


@hydra.main(config_path=str(Path(PLR_ROOT_DIR) / "cfg"), config_name="config.yaml")
def main(cfg: DictConfig):

    if cfg.enable_wandb:
        wandb.login()
        os.makedirs(os.path.join(PLR_ROOT_DIR, "results"), exist_ok=True)

        run = wandb.init(
            name="MNIST_CNN_Run_001",
            dir=os.path.join(PLR_ROOT_DIR, "results"),
            project=cfg.wandb_project,
            settings=wandb.Settings(code_dir=PLR_ROOT_DIR),
            config=cfg,
        )

        def include_fn(path, root):
            return path.endswith(".py") or path.endswith(".yaml")

        run.log_code(name="source_files", root=PLR_ROOT_DIR, include_fn=include_fn)
    else:
        wandb.init(mode="disabled")

    study = optuna.create_study(direction="minimize")

    def wrapped_objective(trial):
        return objective(trial, config=cfg)

    study.optimize(wrapped_objective, n_trials=cfg.n_trials)

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
