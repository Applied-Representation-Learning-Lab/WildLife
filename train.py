import shutil
import pickle
import torch
import timm
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import v2
from lightly.utils.benchmarking import LinearClassifier
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.scheduler import CosineWarmupScheduler
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from torch.optim import SGD, Optimizer
from typing import Any, Dict, List, Tuple, Union
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import pandas as pd
import open_clip
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from lightly.models.utils import deactivate_requires_grad
from lightly.transforms import SimCLRTransform
from efficientnet.model import EfficientNet
from sklearn.metrics import confusion_matrix
import wandb
import onnx
from onnx2pytorch import ConvertModel

from pytorch_lightning import seed_everything
from torchvision.datasets import ImageFolder
from wildlife.utils import get_transform


def train(config):
    seed_everything(config.seed)

    TRAIN_TRANSFORM, TEST_TRANSFORM = get_transform(config)
    train_dataset = ImageFolder(f"data/{config.dataset}/train", transform=TRAIN_TRANSFORM)
    val_dataset = ImageFolder(f"data/{config.dataset}/val", transform=TEST_TRANSFORM)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=args.num_workers)

    backbone = timm.create_model(
        "vit_base_patch16_224.orig_in21k_ft_in1k", 
        pretrained=True
    )
    backbone.head = torch.nn.Identity()

    model = Classifier(model=backbone,
                            batch_size_per_device=batch_size,
                            feature_dim=feature_dim,
                            num_classes=num_classes,
                            class_names=["wildebeest", "wildpig", "elephant", 
                                         "wilddog", "buffalo", "lion", "hyena", 
                                         "rhino", "leopard", "hippo"],
                            freeze_model=args.freeze_model,
                            weight_decay=5e-6,
                            lr=0.002,
                            topk=(1, 2),
                            mlp=args.mlp)

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
                            monitor="val_top1",
                            filename="val_top1",
                            mode="max",
                            save_top_k=1,
                            verbose=True
                        )
    callbacks.append(checkpoint_callback)

    wandb_logger = WandbLogger(
        name=f"{args.model_name}_{config.dataset}", 
        log_model=False, 
        project=config.wildlife
    )

    trainer = pl.Trainer(
        max_epochs=args.train_epochs, 
        accelerator=args.accelerator,
        devices=args.gpu_devices,
        enable_checkpointing=False,
        logger=wandb_logger,
        strategy=args.strategy,
        precision=args.precision,
        sync_batchnorm=args.sync_batchnorm,
    )
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Classifier model")

    parser.add_argument("--config", type=str, help="Name of config file")
    args, remaining_args = parser.parse_known_args()

    config_name = args.config.lower()
    with open(f"config/{config_name}.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    # Add all config parameters as optional arguments
    for key, value in config.items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", type=bool, default=value)
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    
    # Parse again with full argument list
    args = parser.parse_args(remaining_args)
    
    train(args)