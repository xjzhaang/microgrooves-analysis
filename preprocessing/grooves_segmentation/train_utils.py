from pathlib import Path

import torch
from tqdm import tqdm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete, Compose

class Trainer():

    def __init__(self, train_loader, val_loader, model, optimizer, scheduler, scaler, device, save_path, writer, epochs):
        self.dataloader_train = train_loader
        self.dataloader_val = val_loader
        self.model = model.to(device)
        self.loss_function = DiceLoss(sigmoid=True)
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.device = device
        self.epochs = epochs
        self.writer = writer

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs_run = 0
        self.train_loss_ls = []
        self.test_loss_ls = []
        self.best_loss = 999
        self.scaler = scaler
        self.save_path = save_path
        self.post_pred = Compose([AsDiscrete(threshold=0.5)])

    def train(self):

        Path(self.save_path).mkdir(parents=True, exist_ok=True)

        if (Path(self.save_path) / "model.pth").is_file():
            ckpt = torch.load(Path(self.save_path) / "model.pth", map_location="cpu")
            self.epochs_run = ckpt["epoch"]
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.scaler.load_state_dict(ckpt["scaler"])
            self.epochs_run = ckpt["epochs_run"]

        for epoch in range(self.epochs):
            train_loss, lr = self.train_step()
            test_loss = self.validation_step()
            self.epochs_run += 1

            state = dict(
                epoch=epoch + 1,
                model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(),
                epochs_run=self.epochs_run,
                scheduler=self.scheduler.state_dict(),
                scaler=self.scaler.state_dict(),
            )
            torch.save(state, Path(self.save_path) / "model.pth")

            if self.best_loss > test_loss:
                self.best_loss = test_loss
                torch.save(state, Path(self.save_path) / f"best.pth")

    def train_step(self):
        loss = 0
        self.model.train()
        progress_bar = tqdm(enumerate(self.dataloader_train), total=len(self.dataloader_train),
                            desc=f"Epoch #{self.epochs_run}")
        for batch, data in progress_bar:
            X, y = data["image"].to(self.device), data["mask"].astype(torch.int).to(self.device)
            y_pred = self.model(X)
            loss_batch = self.loss_function(y_pred, y)
            loss += loss_batch.item()
            self.scaler.scale(loss_batch).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            lr = self.scheduler.get_last_lr()[0]

            progress_bar.set_postfix_str(
                f"Current loss {loss_batch.item():.5f} ---- Train loss {loss / (batch + 1):.5f} ---- Learning rate {lr:.5f}")
        self.scheduler.step()
        loss = loss / len(self.dataloader_train)
        self.writer.add_scalar('Loss/train', loss, self.epochs_run)
        self.writer.add_scalar('LR/LR', lr, self.epochs_run)

        return loss, lr

    def validation_step(self):

        loss = 0
        self.model.eval()
        with torch.inference_mode():
            progress_bar = tqdm(enumerate(self.dataloader_val), total=len(self.dataloader_val),
                                desc=f"Epoch #{self.epochs_run}")
            for batch, data in progress_bar:
                X, y = data["image"].to(self.device), data["mask"].to(self.device)

                y_pred = sliding_window_inference(X, roi_size=(512, 512), sw_batch_size=4, predictor=self.model, overlap=0.7)
                loss_batch = self.loss_function(y_pred, y)

                loss += loss_batch.item()
                lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix_str(
                    f"Current loss {loss_batch.item():.5f} ---- Val loss {loss / (batch + 1):.5f} ---- Learning rate {lr:.5f}")
            loss = loss / len(self.dataloader_val)
            self.writer.add_scalar('Loss/val', loss, self.epochs_run)
        return loss
