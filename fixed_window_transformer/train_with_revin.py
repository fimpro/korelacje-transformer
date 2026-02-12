#!/usr/bin/env python3
"""Train script extracted from revin-changes.ipynb.

Usage: python train_with_revin.py --root PATH
"""
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import trange, tqdm
import random

from models import revinTransformer


class TimeSeriesDatasetWithMemory(Dataset):
    def __init__(self, X: List[torch.Tensor], Y: List[torch.Tensor]):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_segments_from_files(
    infant_file_path: Path, adult_file_path: Path, min_length: int = 125
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
    stride: int = 10,
    with_long_memory: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X = []
    Y = []
    for segment in segments:
        data = segment[
            ["AU06_r_infant", "AU12_r_infant", "AU06_r_adult", "AU12_r_adult"]
        ].to_numpy()
        num_samples = len(data) - input_length - output_length + 1
        for i in range(0, num_samples, stride):
            if with_long_memory:
                X.append(data[: i + input_length])
            else:
                X.append(data[i : i + input_length])
            Y.append(data[i + input_length : i + input_length + output_length])
    if not with_long_memory:
        numpy_X = np.array(X, dtype=np.float32)
        numpy_Y = np.array(Y, dtype=np.float32)
        split_idx = int(len(X) * train_test_split)
        train_X = torch.from_numpy(numpy_X[:split_idx])
        train_Y = torch.from_numpy(numpy_Y[:split_idx])
        test_X = torch.from_numpy(numpy_X[split_idx:])
        test_Y = torch.from_numpy(numpy_Y[split_idx:])
    else:
        split_idx = int(len(X) * train_test_split)
        train_X = [torch.from_numpy(x.astype(np.float32)) for x in X[:split_idx]]
        train_Y = [torch.from_numpy(y.astype(np.float32)) for y in Y[:split_idx]]
        test_X = [torch.from_numpy(x.astype(np.float32)) for x in X[split_idx:]]
        test_Y = [torch.from_numpy(y.astype(np.float32)) for y in Y[split_idx:]]
    return train_X, train_Y, test_X, test_Y


def load_multiple_files(
    month: int,
    root: Path,
    input_length: int,
    output_length: int,
    with_long_memory: bool = False,
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
                segments,
                input_length=input_length,
                output_length=output_length,
                with_long_memory=with_long_memory,
            )
            if len(tX) == 0:
                continue
            train_X_list.append(tX)
            train_Y_list.append(tY)
            test_X_list.append(eX)
            test_Y_list.append(eY)
    train_X = sum(train_X_list, []) if with_long_memory else torch.cat(train_X_list, dim=0)
    train_Y = sum(train_Y_list, []) if with_long_memory else torch.cat(train_Y_list, dim=0)
    test_X = sum(test_X_list, []) if with_long_memory else torch.cat(test_X_list, dim=0)
    test_Y = sum(test_Y_list, []) if with_long_memory else torch.cat(test_Y_list, dim=0)
    return train_X, train_Y, test_X, test_Y


def sample_mask_flags() -> Tuple[bool, bool]:
    r = torch.rand(1).item()
    if r < 0.1:
        return True, False  # mask mom only (mask_mom, mask_child)
    if r < 0.2:
        return False, True  # mask child only
    return False, False


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


def collate_pad(batch, pad_value: float = 0.0):
    Xs, Ys = zip(*batch)
    Xs = [torch.as_tensor(x, dtype=torch.float32) for x in Xs]
    Ys = [torch.as_tensor(y, dtype=torch.float32) for y in Ys]
    lengths_x = torch.tensor([x.shape[0] for x in Xs], dtype=torch.long)
    lengths_y = torch.tensor([y.shape[0] for y in Ys], dtype=torch.long)
    Xp = pad_sequence(Xs, batch_first=True, padding_value=pad_value)
    Yp = pad_sequence(Ys, batch_first=True, padding_value=pad_value)
    src_key_padding_mask = torch.arange(Xp.size(1))[None, :] >= lengths_x[:, None]
    tgt_key_padding_mask = torch.arange(Yp.size(1))[None, :] >= lengths_y[:, None]
    return Xp, Yp, lengths_x, lengths_y, src_key_padding_mask, tgt_key_padding_mask


def train(
    root: Path,
    month: int = 1,
    input_length: int = 50,
    output_length: int = 10,
    batch_size: int = 32,
    epochs: int = 5,
    lr: float = 1e-4,
    with_long_memory: bool = True,
    save_path: Path = Path("./revin_transformer_model.pth"),
    best_path: Path = Path("./revin_transformer_model_best.pth"),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_X, train_Y, test_X, test_Y = load_multiple_files(
        month, root, input_length=input_length, output_length=output_length, with_long_memory=with_long_memory
    )

    train_loader = DataLoader(
        TimeSeriesDatasetWithMemory(train_X, train_Y),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_pad,
    )
    test_loader = DataLoader(
        TimeSeriesDatasetWithMemory(test_X, test_Y),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_pad,
    )

    model = revinTransformer.RevinTransformer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    patience = 8
    best_val = float("inf")
    no_improve = 0
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in trange(epochs, desc="epochs"):
        model.train()
        total_loss = 0.0
        for Xp, Yp, len_x, len_y, src_mask, tgt_mask in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            Xp = Xp.to(device)
            Yp = Yp.to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)
            child_src = Xp[..., :2]
            mom_src = Xp[..., 2:]
            tgt_in = torch.zeros_like(Yp)
            tgt_in[:, 1:, :] = Yp[:, :-1, :]
            child_tgt_in = tgt_in[..., :2]
            mom_tgt_in = tgt_in[..., 2:]
            mask_mom, mask_child = sample_mask_flags()
            optimizer.zero_grad()
            pred = model(
                child_src,
                mom_src,
                child_tgt_in,
                mom_tgt_in,
                mask_child=mask_child,
                mask_mom=mask_mom,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_mask,
            )
            loss = masked_mse_loss(pred, Yp, mask_child=mask_child, mask_mom=mask_mom)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xp.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_src, val_tgt, val_len_x, val_len_y, val_src_pad_mask, val_tgt_pad_mask in tqdm(test_loader, desc=f"val epoch {epoch}", leave=False):
                val_src = val_src.to(device)
                val_tgt = val_tgt.to(device)
                val_src_pad_mask = val_src_pad_mask.to(device)
                val_tgt_pad_mask = val_tgt_pad_mask.to(device)
                val_child_src = val_src[..., :2]
                val_mom_src = val_src[..., 2:]
                val_tgt_in = torch.zeros_like(val_tgt)
                val_tgt_in[:, 1:, :] = val_tgt[:, :-1, :]
                val_child_tgt_in = val_tgt_in[..., :2]
                val_mom_tgt_in = val_tgt_in[..., 2:]
                val_pred = model(
                    val_child_src,
                    val_mom_src,
                    val_child_tgt_in,
                    val_mom_tgt_in,
                    mask_child=False,
                    mask_mom=False,
                    src_key_padding_mask=val_src_pad_mask,
                    tgt_key_padding_mask=val_tgt_pad_mask,
                )
                val_loss += masked_mse_loss(val_pred, val_tgt, mask_child=False, mask_mom=False).item() * val_src.size(0)
        val_loss = val_loss / len(test_loader.dataset)
        print(f"end=epoch {epoch}: train_loss={avg_loss:.6f}, val_loss={val_loss:.6f}")
        scheduler.step()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            no_improve = 0
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict()}, best_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, save_path)


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=False, default=Path("/net/tscratch/people/plgfimpro/korelacje/short_fixed_results_openface2"))
    p.add_argument("--month", type=int, default=1)
    p.add_argument("--input-length", type=int, default=50)
    p.add_argument("--output-length", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-path", type=Path, default=Path("./revin_transformer_model.pth"))
    p.add_argument("--best-path", type=Path, default=Path("./revin_transformer_model_best.pth"))
    p.add_argument("--no-long-memory", dest="with_long_memory", action="store_false")
    p.set_defaults(with_long_memory=True)
    args = p.parse_args()
    train(
        args.root,
        month=args.month,
        input_length=args.input_length,
        output_length=args.output_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        with_long_memory=args.with_long_memory,
        save_path=args.save_path,
        best_path=args.best_path,
    )


if __name__ == "__main__":
    cli()
