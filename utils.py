import atexit
import shutil
import os
import mlflow
from tensorboardX import SummaryWriter
from pathlib import Path

from torch.optim.lr_scheduler import ReduceLROnPlateau

writer: SummaryWriter = None
e: int = None
train_iter: int = 0
val_iter: int = 0
test_iter: int = 0
global_count: int = 0

class CustomLogger(SummaryWriter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        mlflow.log_param("tfboard", self.log_dir)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        super(CustomLogger, self).add_scalar(tag, scalar_value, global_step, walltime)
        if 'epoch' in tag.lower():
            if '/' in tag.lower():
                tag = tag.split("/")[1]
            mlflow.log_metric(tag, scalar_value)

    def log_model(self, model, tag='Model'):
        self.add_text(tag,
                      repr(model).replace("  ", "&nbsp;&nbsp;").replace("\n", "  \n"))

    def log_args(self, args_dict):
        for key, value in args_dict.items():
            if key in ['dataset', 'batch_size', 'link_pred', 'min_cut', 'pool_size', 'pool_ratio',
                       'ncut', 'gumbel']:
                mlflow.log_param(key, value)

    def log_py_file(self):
        cwd = Path()
        for file in cwd.iterdir():
            if file.suffix == '.py':
                mlflow.log_artifact(str(file.resolve()), artifact_path="pyfiles")

    def log_tfboard(self):
        mlflow.log_artifacts(self.log_dir, artifact_path=self.log_dir)

    def log_epochs(self, e):
        mlflow.log_metric("Epochs", e)


class ConstantScheduler:
    def __init__(self, **kwargs):
        pass

    def step(self, loss):
        pass

def del_tensorboard():
    print(writer.log_dir)
    option = input("Delete?")

    if option.lower() == 'd':
        print(shutil.rmtree(writer.log_dir))
        print("DELETED!")


import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


class BestRecorder:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, path="./"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.best_acc = 0
        self.path = path
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_acc, model):

        if val_acc >= self.best_acc:
            self.save_checkpoint(val_acc, model)
            self.best_acc = val_acc

    def save_checkpoint(self, val_acc, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation Acc ({self.best_acc:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.path, 'checkpoint.pt'))
