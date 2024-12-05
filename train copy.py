import os
import torch
import wandb
from tqdm import tqdm, trange
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from data.data_loader import loader
from utils.Loss import Dice_CE_Loss, Topological_Loss
from augmentation.Augmentation import Cutout, cutmix
from wandb_init import parser_init, wandb_init
import yaml
from utils.metrics import *

from models.Model import model_dice_bce

# Utility Functions
def load_deeplabv3(num_classes):
    """Load the DeepLabV3 model and adjust for the dataset."""
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model


def load_config(config_name):
    """Load a YAML configuration file."""
    with open(config_name) as file:
        return yaml.safe_load(file)


def using_device():
    """Set and print the device used for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    return device


def setup_paths(data):
    """Set up data paths for training and validation."""
    folder_mapping = {
        "isic_1": "isic_1/",
        "kvasir_1": "kvasir_1/",
        "ham_1": "ham_1/",
        "PH2Dataset": "PH2Dataset/",
        "isic_2016_1": "isic_2016_1/"
    }
    folder = folder_mapping.get(data, "isic_2016_1/")
    base_path = os.environ["ML_DATA_OUTPUT"] if torch.cuda.is_available() else os.environ["ML_DATA_OUTPUT_LOCAL"]
    return os.path.join(base_path, folder)


# Main Function
def main():
    # Configuration and Initial Setup
    data, training_mode, train, addtopoloss, aug_reg = 'isic_2016_1', "supervised", True, True, False
    aug_threshould, best_valid_loss = 0, float("inf")
    device = using_device()
    folder_path = setup_paths(data)
    args, res, config_res = parser_init("segmentation task", "training", training_mode, train)
    config = wandb_init(os.environ["WANDB_API_KEY"], os.environ["WANDB_DIR"], args, ', '.join(config_res), data)

    # Data Loaders
    def create_loader(aug):
        return loader(args.mode, args.sslmode_modelname, args.train, args.bsize, args.workers,
                      args.imsize, args.cutoutpr, args.cutoutbox, aug, args.shuffle, args.sratio, data)

    train_loader = create_loader(args.aug)
    val_loader = create_loader(False)

    # Model, Loss, Optimizer, Scheduler
    num_classes = config['n_classes']
    model = model_dice_bce(num_classes, config_res, args.mode, args.imnetpr).to(device)

    optimizer = Adam(model.parameters(), lr=config['learningrate'])
    scheduler = CosineAnnealingLR(optimizer, config['epochs'], eta_min=config['learningrate'] / 10)
    loss_fn, topo_loss_fn = Dice_CE_Loss(), Topological_Loss(lam=0.1).to(device)

    print(f"Training on {len(train_loader) * args.bsize} images. Saving checkpoints to {folder_path}")

    # Training and Validation Loops
    def run_epoch(loader, training=True):
        """Run a single training or validation epoch."""
        epoch_loss, epoch_dice_loss, epoch_topo_loss = 0.0, 0.0, 0.0
        model.train() if training else model.eval()

        with torch.set_grad_enabled(training):
            for images, labels in tqdm(loader, desc="Training" if training else "Validating", leave=False):
                images, labels = images.to(device), labels.to(device)

                # Apply augmentations during training
                if training and args.aug:
                    images, labels = cutmix(images, labels, args.cutmixpr)
                    images, labels = Cutout(images, labels, args.cutoutpr, args.cutoutbox)

                _, out = model(images)
                dice_loss = loss_fn.Dice_BCE_Loss(out, labels)

                if addtopoloss:
                    topo_loss = topo_loss_fn(out, labels)
                    total_loss = dice_loss + topo_loss
                    epoch_topo_loss += topo_loss.item()
                else:
                    total_loss = dice_loss

                epoch_loss += total_loss.item()
                epoch_dice_loss += dice_loss.item()

                if training:
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

        return epoch_loss / len(loader), epoch_dice_loss / len(loader), epoch_topo_loss / len(loader)

    for epoch in trange(config['epochs'], desc="Epochs"):
        # Reduce augmentation probability
        if aug_reg and aug_threshould / 2 <= epoch <= aug_threshould:
            args.cutoutpr -= epoch / (aug_threshould * 4)
            args.cutmixpr -= epoch / (aug_threshould * 4)

        # Training
        train_loss, train_dice_loss, train_topo_loss = run_epoch(train_loader, training=True)
        wandb.log({"Train Loss": train_loss, "Train Dice Loss": train_dice_loss, "Train Topo Loss": train_topo_loss})

        # Validation
        val_loss, val_dice_loss, val_topo_loss = run_epoch(val_loader, training=False)
        wandb.log({"Val Loss": val_loss, "Val Dice Loss": val_dice_loss, "Val Topo Loss": val_topo_loss})

        # Print losses for this epoch
        print(f"Epoch {epoch + 1}/{config['epochs']} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Dice Loss: {train_dice_loss:.4f}, Val Dice Loss: {val_dice_loss:.4f}, "
              f"Train Topo Loss: {train_topo_loss:.4f}, Val Topo Loss: {val_topo_loss:.4f}")

        # Save best model
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), os.path.join(folder_path, f"{model.__class__.__name__}.pt"))
            print(f"Best model saved with Val Loss: {val_loss:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()
