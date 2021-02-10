import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import LTBC
from data import places365DataModule


def main(args):
    # Settings
    pl.seed_everything(42)
    
    # Handle the data
    dm = places365DataModule(args.data_folder, batch_size=args.batch_size)

    # Define model
    model = LTBC(alpha=args.alpha, lr=args.lr, rightsize=True, classify=True)

    # Exp logger
    logger = TensorBoardLogger('logs/tensorboard_logs')

    # Define training
    checkpointer = ModelCheckpoint(monitor='val_loss',
                                   save_top_k=3,
                                   mode='min',
                                   save_last=True,
                                   filename='{epoch}-{val_loss:.2f}-{train_loss:.2f}')
    trainer = pl.Trainer(gpus=1,
                         num_nodes=args.n_nodes,
                         accelerator='ddp',
                         auto_select_gpus=True,
                         max_epochs=args.epochs,
                         callbacks=[checkpointer],
                         logger=logger,
                         limit_train_batches=0.2,
                         limit_val_batches=0.2,
                         limit_test_batches=0.2,
                         val_check_interval=0.5,
                         auto_scale_batch_size='binsearch')

    # Train
    trainer.fit(model, dm)


if __name__ ==  '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n_nodes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=0.1)

    parser.add_argument('--data_folder', type=str, default="../places365_standard")
    args = parser.parse_args()
    main(args)
