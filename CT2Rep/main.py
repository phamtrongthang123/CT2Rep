import numpy as np
import pytorch_lightning as pl
import torch
from modules.data_ct import CTReportDataset
from modules.tokenizers import Tokenizer
from modules.trainer import CT2RepLightningModule
from pytorch_lightning.cli import LightningCLI
from torch.utils.data import DataLoader


def custom_collate_fn(data):
    images_id, images, reports_ids, reports_masks, seq_lengths = zip(*data)
    images = torch.stack(images, 0)
    max_seq_length = max(seq_lengths)
    targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
    targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

    for i, report_ids in enumerate(reports_ids):
        targets[i, : len(report_ids)] = report_ids

    for i, report_masks in enumerate(reports_masks):
        targets_masks[i, : len(report_masks)] = report_masks

    return (
        images_id,
        images,
        torch.LongTensor(targets),
        torch.FloatTensor(targets_masks),
    )


class CTReportDataModule(pl.LightningDataModule):
    def __init__(
        self,
        trainfolder: str,
        validfolder: str,
        xlsxfile: str,
        max_seq_length: int,
        tokenizer_config: dict,
        num_frames: int = 2,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["tokenizer"])
        self.tokenizer = Tokenizer(**tokenizer_config)

        # Store parameters
        self.trainfolder = trainfolder
        self.validfolder = validfolder
        self.xlsxfile = xlsxfile
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_length = max_seq_length
        # Will hold datasets
        self.train_ds = None
        self.valid_ds = None

    def setup(self, stage=None):
        """Called on every process in DDP"""
        self.train_ds = CTReportDataset(
            max_seq_length=self.max_seq_length,
            data_folder=self.trainfolder,
            xlsx_file=self.xlsxfile,
            tokenizer=self.tokenizer,
            num_frames=self.num_frames,
        )
        self.valid_ds = CTReportDataset(
            max_seq_length=self.max_seq_length,
            data_folder=self.validfolder,
            xlsx_file=self.xlsxfile,
            tokenizer=self.tokenizer,
            num_frames=self.num_frames,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            shuffle=True,
            collate_fn=custom_collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_ds,
            shuffle=False,
            collate_fn=custom_collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Add tokenizer configuration
        parser.add_class_arguments(Tokenizer, "tokenizer")
        parser.add_argument(
            "--xlsxfile",
            type=str,
            required=True,
            help="Path to the Excel file used by all components",
        )

    def instantiate_classes(self):
        # the ${} doesn't work in this version so i have to set manually
        self.config.fit.tokenizer.xlsxfile = self.config.fit.xlsxfile
        # Inject into model and datamodule configs
        self.config.fit.data.tokenizer_config = dict(self.config.fit.tokenizer)
        # breakpoint()
        self.config.fit.model.tokenizer_config = dict(self.config.fit.tokenizer)

        # other mapping
        self.config.fit.data.max_seq_length = (
            self.config.fit.model.encoder_decoder_config["max_seq_length"]
        )
        self.config.fit.data.xlsxfile = self.config.fit.xlsxfile
        self.config.fit.trainer.default_root_dir = self.config.fit.model.save_dir
        self.config.fit.trainer.callbacks[
            0
        ].init_args.dirpath = self.config.fit.model.save_dir
        self.config.fit.trainer.callbacks[
            0
        ].init_args.monitor = f"val_{self.config.fit.model.monitor_metric}"
        self.config.fit.trainer.callbacks[0].init_args.filename = (
            "best-{epoch}-{val_" + self.config.fit.model.monitor_metric + ":.2f}"
        )
        super().instantiate_classes()


if __name__ == "__main__":
    cli = CustomCLI(CT2RepLightningModule, CTReportDataModule)
