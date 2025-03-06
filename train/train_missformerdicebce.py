import os
import torch
import wandb
from tqdm import tqdm, trange
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from data.data_loader import loader
from utils.Loss import Dice_CE_Loss
from augmentation.Augmentation import Cutout, cutmix
from wandb_init import parser_init, wandb_init
import yaml
from utils.metrics import calculate_metrics
#from models.Model4 import model_bce_topo      #256
#from models.FAT_NET import FAT_Net          #224
from models.MISSFormer import MISSFormer    #224

def load_deeplabv3(num_classes):             #256
    """Load the DeepLabV3 model and adjust for the dataset."""
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model

def process_model_output(model, images):
    """Process the output of the model based on its type."""
    model_type = model.__class__.__name__
    if model_type == 'DeepLabV3':
        out = model(images)
        return out['out']
    # Default case
    return model(images)

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
        "isic_2018_1": "isic_1/",
        "kvasir_1": "kvasir_1/",
        "ham_1": "ham_1/",
        "PH2Dataset": "PH2Dataset/",
        "isic_2016_1": "isic_2016_1/"
    }
    folder = folder_mapping.get(data)
    base_path = os.environ["ML_DATA_OUTPUT"] if torch.cuda.is_available() else os.environ["ML_DATA_OUTPUT_LOCAL"]
    return os.path.join(base_path, folder)


# Main Function
def main():
    # Configuration and Initial Setup
    data, training_mode, train, addtopoloss, aug_reg = 'PH2Dataset', "supervised", True, False, False
    aug_threshould, best_valid_loss = 0, float("inf")
    device = using_device()
    folder_path = setup_paths(data)
    args, res = parser_init("segmentation task", "training", training_mode, train)
    config = wandb_init(os.environ["WANDB_API_KEY"], os.environ["WANDB_DIR"], args, data)
    
    # Data Loaders
    def create_loader(aug):
        return loader(args.mode, args.sslmode_modelname, args.train, args.bsize, args.workers,
                      args.imsize, args.cutoutpr, args.cutoutbox, aug, args.shuffle, args.sratio, data)

    train_loader = create_loader(args.aug)
    val_loader = create_loader(False)

    # Model, Loss, Optimizer, Scheduler
    num_classes = config['n_classes']
    model = MISSFormer().to(device)
    #model = load_deeplabv3(num_classes).to(device)
    #model = FAT_Net().to(device)
    #model = MISSFormer().to(device)
    checkpoint_path = folder_path+str(model.__class__.__name__)+str(res)
    optimizer = Adam(model.parameters(), lr=config['learningrate'])
    scheduler = CosineAnnealingLR(optimizer, config['epochs'], eta_min=config['learningrate'] / 10)
    loss_fn = Dice_CE_Loss()
    
    if addtopoloss:
        from utils.Loss import Topological_Loss
        topo_loss_fn = Topological_Loss(lam=0.1).to(device)

    from ptflops import get_model_complexity_info
    import re

    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
    print_per_layer_stat=True, verbose=True)
    flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
    flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
    print('Computational complexity: {:<8}'.format(macs))
    print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
    print('Number of parameters: {:<8}'.format(params))

    print(f"Training on {len(train_loader) * args.bsize} images. Saving checkpoints to {folder_path}")
    print('Train loader transform',train_loader.dataset.tr)
    print('Val loader transform',val_loader.dataset.tr)
    print(f"model config : {checkpoint_path}")
    
    
    # Training and Validation Loops
    def run_epoch(loader, training=True):
        """Run a single training or validation epoch."""
        epoch_loss, epoch_loss_, epoch_topo_loss = 0.0, 0.0, 0.0
        model.train() if training else model.eval()

        val_metrics = [0.0] * 5 
        metrics_sum = [0.0] * 5  # To sum up metrics
        num_batches = 0
        with torch.set_grad_enabled(training):
            for images, labels in tqdm(loader, desc="Training" if training else "Validating", leave=False):
                images, labels = images.to(device), labels.to(device)

                # Apply augmentations during training
                if training and args.aug:
                    images, labels = cutmix(images, labels, args.cutmixpr)
                    images, labels = Cutout(images, labels, args.cutoutpr, args.cutoutbox)
                
                out = process_model_output(model,images)

                loss_ = loss_fn.Dice_BCE_Loss(out, labels)

                if addtopoloss:
                    from utils.Loss import Topological_Loss
                    topo_loss = topo_loss_fn(out, labels)
                    total_loss = loss_ + topo_loss
                    epoch_topo_loss += topo_loss.item()
                else:
                    total_loss = loss_

                epoch_loss += total_loss.item()
                epoch_loss_ += loss_.item()

                # Calculate metrics during validation
                if not training:
                    prediction = (torch.sigmoid(out) > 0.5).float()
                    batch_metrics = calculate_metrics(labels.cpu(), prediction.cpu())
                    metrics_sum = [x + y for x, y in zip(metrics_sum, batch_metrics)]
                    num_batches += 1

                if training:
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    scheduler.step()

        if not training and num_batches > 0:
            val_metrics = [x / num_batches for x in metrics_sum]
            return epoch_loss / len(loader), epoch_loss_ / len(loader), epoch_topo_loss / len(loader), val_metrics

        if not training:
            return epoch_loss / len(loader), epoch_loss_ / len(loader), epoch_topo_loss / len(loader), val_metrics

        return epoch_loss / len(loader), epoch_loss_ / len(loader), epoch_topo_loss / len(loader)

    for epoch in trange(config['epochs'], desc="Epochs"):
        # Reduce augmentation probability
        if aug_reg and aug_threshould / 2 <= epoch <= aug_threshould:
            args.cutoutpr -= epoch / (aug_threshould * 4)
            args.cutmixpr -= epoch / (aug_threshould * 4)

        # Training
        train_loss, train_loss_, train_topo_loss = run_epoch(train_loader, training=True)
        wandb.log({"Train Loss": train_loss, "Train Dice Loss": train_loss_, "Train Topo Loss": train_topo_loss})

        # Validation
        if epoch == 0:
            # Compute validation losses but set metrics to zero
            val_loss, val_loss_, val_topo_loss, _ = run_epoch(val_loader, training=False)
            val_metrics = [0.0] * 5  # Set metrics to zero
        else:
            val_loss, val_loss_, val_topo_loss, val_metrics = run_epoch(val_loader, training=False)
            
        wandb.log({
            "Val Loss": val_loss,
            "Val Dice Loss": val_loss_,
            "Val Topo Loss": val_topo_loss,
            "Val IoU": val_metrics[0],
            "Val Dice": val_metrics[1],
            "Val Recall": val_metrics[2],
            "Val Precision": val_metrics[3],
            "Val Accuracy": val_metrics[4],
        })

        # Print losses and validation metrics
        print(f"Epoch {epoch + 1}/{config['epochs']} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Dice Loss: {train_loss_:.4f}, Val Dice Loss: {val_loss_:.4f}, "
              f"Train Topo Loss: {train_topo_loss:.4f}, Val Topo Loss: {val_topo_loss:.4f}")
        
        print(f"Validation Metrics: IoU: {val_metrics[0]:.4f}, Dice: {val_metrics[1]:.4f}, "
              f"Recall: {val_metrics[2]:.4f}, Precision: {val_metrics[3]:.4f}, "
              f"Accuracy: {val_metrics[4]:.4f}")

        # Save best model
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved with Val Loss: {val_loss:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()
