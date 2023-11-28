

import torch
from torch.utils.tensorboard import SummaryWriter
from monai.networks.nets import UNet

from train_utils import Trainer
from data_builder import build_dataloaders

model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
)
EPOCH=200
optimizer = torch.optim.Adam(model.parameters(), 1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH)
scaler = torch.cuda.amp.GradScaler()
train_loader, val_loader, _ = build_dataloaders(num_workers=4, batch_size_train=2, batch_size_val=1)
writer = SummaryWriter()
trainer = Trainer(train_loader, val_loader, model, optimizer, scheduler, scaler,
                  device='cuda', save_path="./models", writer=writer, epochs=EPOCH)
trainer.train()
writer.close()