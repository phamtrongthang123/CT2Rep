import csv
import os

import pytorch_lightning as pl


class CT2RepLightningModule(pl.LightningModule):
    """
    Streamlined PyTorch Lightning trainer for CT2Rep model.

    This is a minimal implementation that provides the core Lightning functionality
    while allowing developers to easily customize the training logic.
    """

    def __init__(
        self, model, criterion, metric_ftns, optimizer_fn, lr_scheduler_fn, args
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "model",
                "criterion",
                "metric_ftns",
                "optimizer_fn",
                "lr_scheduler_fn",
            ]
        )

        # Core components
        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer_fn = optimizer_fn
        self.lr_scheduler_fn = lr_scheduler_fn
        self.args = args

        # Initialize validation outputs storage for v2.0 compatibility
        self.validation_outputs = []

        # Create save directory if needed
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    def training_step(self, batch, batch_idx):
        """Training step - customize this method for your specific training logic"""
        images_id, images, reports_ids, reports_masks = batch
        output = self.model(images, reports_ids, mode="train")
        loss = self.criterion(output, reports_ids, reports_masks)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step - customize this method for your specific validation logic"""
        images_id, images, reports_ids, reports_masks = batch
        output = self.model(images, mode="sample")
        reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
        ground_truths = self.model.tokenizer.decode_batch(
            reports_ids[:, 1:].cpu().numpy()
        )

        # Store outputs in instance attribute for v2.0 compatibility
        output_dict = {
            "reports": reports,
            "ground_truths": ground_truths,
            "images_id": images_id,
        }
        self.validation_outputs.append(output_dict)

        return output_dict

    def on_validation_epoch_end(self):
        """Aggregate validation outputs and compute metrics - v2.0 compatible"""
        # Access stored outputs from instance attribute
        outputs = self.validation_outputs

        # Aggregate all validation outputs
        val_res = []
        val_gts = []
        for output in outputs:
            val_res.extend(output["reports"])
            val_gts.extend(output["ground_truths"])

        # Save results to CSV for analysis
        if val_gts and val_res:  # Ensure we have data
            self._save_validation_results(val_res, val_gts)

            # Compute and log metrics
            val_met = self.metric_ftns(
                {i: [gt] for i, gt in enumerate(val_gts)},
                {i: [re] for i, re in enumerate(val_res)},
            )

            for k, v in val_met.items():
                self.log(f"val_{k}", v, prog_bar=True)

        # Clear outputs for next epoch
        self.validation_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = self.optimizer_fn
        scheduler = self.lr_scheduler_fn

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": f"val_{self.args.monitor_metric}",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _save_validation_results(self, val_res, val_gts):
        """Save validation results to CSV files"""
        epoch = self.current_epoch
        dir_save = self.args.save_dir
        gts_file = os.path.join(dir_save, f"{epoch}_gts.csv")
        res_file = os.path.join(dir_save, f"{epoch}_res.csv")

        with (
            open(gts_file, "w", newline="") as gtss,
            open(res_file, "w", newline="") as ress,
        ):
            gt_writer = csv.writer(gtss)
            gen_writer = csv.writer(ress)
            for gt, res in zip(val_gts, val_res):
                gt_writer.writerow([str(gt)])
                gen_writer.writerow([str(res)])

    # Optional: Keep legacy compatibility
    def _train_epoch(self, epoch):
        """
        Legacy training epoch method - kept for backward compatibility.

        Note: When using PyTorch Lightning, prefer using training_step() instead.
        This method is provided for developers who want to migrate gradually.
        """
        print("Warning: _train_epoch is deprecated when using PyTorch Lightning.")
        print("Please use training_step() and validation_step() methods instead.")
        return {}


# Backward compatibility alias
Trainer = CT2RepLightningModule
