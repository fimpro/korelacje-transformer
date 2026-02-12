import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def load_segments_from_files(
    infant_file_path: Path,
    adult_file_path: Path,
    min_length: int = 125,
) -> List[pd.DataFrame]:
    meaningfull_columns = ["Frame", "success", "AU06_r", "AU12_r"]
    adult_file = pd.read_csv(adult_file_path)[meaningfull_columns]
    infant_file = pd.read_csv(infant_file_path)[meaningfull_columns]

    merged_file = pd.merge(
        infant_file,
        adult_file,
        on="Frame",
        how="inner",
        suffixes=("_infant", "_adult"),
    )

    continous_dfs: List[pd.DataFrame] = []
    output_columns = [
        "Frame",
        "AU06_r_infant",
        "AU12_r_infant",
        "AU06_r_adult",
        "AU12_r_adult",
    ]
    current_segment = pd.DataFrame(columns=output_columns)
    for row in merged_file.itertuples():
        if row.success_infant >= 0.75 and row.success_adult >= 0.75:
            current_segment.loc[len(current_segment)] = (
                row.Frame,
                row.AU06_r_infant,
                row.AU12_r_infant,
                row.AU06_r_adult,
                row.AU12_r_adult,
            )
        else:
            if len(current_segment) > min_length:
                continous_dfs.append(current_segment)
            current_segment = pd.DataFrame(columns=output_columns)
    if len(current_segment) > min_length:
        continous_dfs.append(current_segment)
    return continous_dfs


def segments_from_one_video_to_dataset(
    segments: List[pd.DataFrame],
    input_length: int,
    output_length: int,
    train_test_split: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X = []
    Y = []
    for segment in segments:
        data = segment[
            ["AU06_r_infant", "AU12_r_infant", "AU06_r_adult", "AU12_r_adult"]
        ].to_numpy()
        num_samples = len(data) - input_length - output_length + 1
        for i in range(num_samples):
            X.append(data[i : i + input_length])
            Y.append(data[i + input_length : i + input_length + output_length])
    numpy_X = np.array(X, dtype=np.float32)
    numpy_Y = np.array(Y, dtype=np.float32)
    split_idx = int(len(X) * train_test_split)
    train_X = torch.from_numpy(numpy_X[:split_idx])
    train_Y = torch.from_numpy(numpy_Y[:split_idx])
    test_X = torch.from_numpy(numpy_X[split_idx:])
    test_Y = torch.from_numpy(numpy_Y[split_idx:])
    return train_X, train_Y, test_X, test_Y


def load_multiple_files(
    month: int,
    root: Path,
    input_length: int,
    output_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    train_X_list: List[torch.Tensor] = []
    train_Y_list: List[torch.Tensor] = []
    test_X_list: List[torch.Tensor] = []
    test_Y_list: List[torch.Tensor] = []
    for subdir in subdirs:
        if subdir.match(f"**/diti_*_{month}"):
            infant_path = subdir / "Kamera 1" / "infant.csv"
            adult_path = subdir / "Kamera 2" / "adult.csv"
            if not infant_path.exists() or not adult_path.exists():
                continue
            segments = load_segments_from_files(infant_path, adult_path)
            if not segments:
                continue
            tX, tY, eX, eY = segments_from_one_video_to_dataset(
                segments, input_length=input_length, output_length=output_length
            )
            if len(tX) == 0:
                continue
            train_X_list.append(tX)
            train_Y_list.append(tY)
            test_X_list.append(eX)
            test_Y_list.append(eY)
    if not train_X_list:
        raise ValueError("No training data found for the specified month")
    train_X = torch.cat(train_X_list, dim=0)
    train_Y = torch.cat(train_Y_list, dim=0)
    test_X = torch.cat(test_X_list, dim=0)
    test_Y = torch.cat(test_Y_list, dim=0)
    return train_X, train_Y, test_X, test_Y

class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        """
        Args:
            X (torch.Tensor): Input data of shape (num_samples, seq_len, input_dim).
            Y (torch.Tensor): Target data of shape (num_samples, seq_len, output_dim).
        """
        self.X = X
        self.Y = Y

    def __len__(self):
        # Return the number of samples
        return len(self.X)

    def __getitem__(self, idx):
        # Return a single sample (X[idx], Y[idx])
        return self.X[idx], self.Y[idx]

def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask_child: bool,
    mask_mom: bool,
    child_dim: int = 2,
) -> torch.Tensor:
    per_elem = (pred - target).pow(2)

    if mask_child and mask_mom:
        return per_elem.new_tensor(0.0)
    if mask_child:
        per_elem = per_elem[..., child_dim:]
    elif mask_mom:
        per_elem = per_elem[..., :child_dim]

    return per_elem.mean()

def sample_mask_flags():
    r = torch.rand(1).item()
    if r < 0.1:
        return True, False  # mask mom only
    if r < 0.2:
        return False, True  # mask child only
    return False, False

class EarlyStopping:
    """Stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, min_delta=0.0, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.path)

def plot_long_segment(start: int = 8500, timestep = 24, mask_mom = False, mask_child = False, dataset=None, model=None,
                       device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    num_examples = 1000
    true = []
    pred = []
    for example_idx in range(num_examples):
        rand_idx = start + example_idx
        src_single, tgt_single = dataset[rand_idx]
        src_single = src_single.unsqueeze(0).to(device)
        tgt_single = tgt_single.unsqueeze(0).to(device)
        
        child_src = src_single[..., :2]
        mom_src = src_single[..., 2:]
        # Use autoregressive generation instead of passing zeros
        pred_single = model.forward_autoregressive(child_src, mom_src, 
                                    mask_child=mask_child, mask_mom=mask_mom, tgt_len=tgt_single.size(1))
        #print(tgt_single.shape)
        #print(pred_single.shape)
        true.append(tgt_single[0,timestep].cpu().numpy())
        pred.append(pred_single[0, timestep].cpu().detach().numpy())
    print("MSE:", np.mean((np.array(true) - np.array(pred))**2, axis=0))
    fig, axes = plt.subplots(4, figsize=(16, 16))
    labels = ['AU06_infant', 'AU12_infant', 'AU06_adult', 'AU12_adult']
    for feat_idx, label in enumerate(labels):
        ax = axes[feat_idx]
        true_feat = [t[feat_idx] for t in true]
        pred_feat = [p[feat_idx] for p in pred]
        ax.plot(pred_feat, label='pred', linewidth=2)
        ax.plot(true_feat, label='true', linewidth=2, linestyle='--')
        ax.set_title(f"{label} over long segment, mse = {np.mean((np.array(true_feat) - np.array(pred_feat))**2):.6f}" + " with masked mom" if mask_mom else "" + " and masked child" if mask_child else "")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_MSE(model, start: int = 8500, timestep = 24, mask_mom = False, mask_child = False, dataset=None, num_examples: int = 500, device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    if dataset is None:
        raise ValueError("dataset must be provided")
    if start < 0:
        start = 0
    max_available = max(0, len(dataset) - start)
    num_examples = min(num_examples, max_available)
    if num_examples == 0:
        raise ValueError("dataset is too small for the requested start/num_examples")
    if timestep < 0:
        raise ValueError("timestep must be non-negative")
    true = []
    pred = []
    for example_idx in range(num_examples):
        rand_idx = start + example_idx
        src_single, tgt_single = dataset[rand_idx]
        src_single = src_single.unsqueeze(0).to(device)
        tgt_single = tgt_single.unsqueeze(0).to(device)
        
        child_src = src_single[..., :2]
        mom_src = src_single[..., 2:]
        # Use autoregressive generation instead of passing zeros
        pred_single = model.forward_autoregressive(child_src, mom_src, 
                                    mask_child=mask_child, mask_mom=mask_mom, tgt_len=tgt_single.size(1))
        #print(tgt_single.shape)
        #print(pred_single.shape)
        true.append(tgt_single[0,timestep].cpu().numpy())
        pred.append(pred_single[0, timestep].cpu().detach().numpy())
    return np.mean((np.array(true) - np.array(pred))**2, axis=0)