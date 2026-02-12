import argparse
import math
from pathlib import Path
import math
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from models.maskedTimeSeriesTransformerWithHistory import MaskedTimeSeriesTransformerWithHistory
from utils import *

class TimeSeriesDataset(Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.X = X
        self.Y = Y

    def __len__(self):
        # Return the number of samples
        return len(self.X)

    def __getitem__(self, idx):
        # Return a single sample (X[idx], Y[idx])
        return self.X[idx], self.Y[idx]
        
def sample_mask_flags() -> Tuple[bool, bool]:
    r = torch.rand(1).item()
    if r < 0.1:
        return True, False  # mask mom only
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
    """Compute MSE while ignoring channels of the masked person.

    Assumes last dimension is feature/channel dim, with:
    - child in [:child_dim]
    - mom   in [child_dim:]

    The model still predicts full `pred`; we only change what contributes to loss.
    """
    per_elem = (pred - target).pow(2)  # same shape as pred/target

    if mask_child and mask_mom:
        return per_elem.new_tensor(0.0)
    if mask_child:
        per_elem = per_elem[..., child_dim:]
    elif mask_mom:
        per_elem = per_elem[..., :child_dim]

    return per_elem.mean()


def train_masked_model(
    month: int,
    data_root: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    input_length: int,
    output_length: int,
    device: torch.device,
    save_path: Path,
) -> None:
    train_X, train_Y, test_X, test_Y = load_multiple_files(
        month, data_root, input_length, output_length
    )
    train_loader = DataLoader(TimeSeriesDataset(train_X, train_Y), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TimeSeriesDataset(test_X, test_Y), batch_size=batch_size, shuffle=False)

    model = MaskedTimeSeriesTransformerWithHistory(d_model=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
    )

    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=7, path='best_model.pt')

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_loss = 0.0
        
        for batch_src, batch_tgt in train_loader:
            batch_src = batch_src.to(device)
            batch_tgt = batch_tgt.to(device)

            # Prepare inputs
            child_src = batch_src[..., :2]
            mom_src = batch_src[..., 2:]
            
            tgt_in = torch.zeros_like(batch_tgt)
            tgt_in[:, 1:, :] = batch_tgt[:, :-1, :]
            child_tgt_in = tgt_in[..., :2]
            mom_tgt_in = tgt_in[..., 2:]

            mask_mom, mask_child = sample_mask_flags()

            optimizer.zero_grad()
            
            pred = model(
                child_src, mom_src, 
                child_tgt_in, mom_tgt_in, 
                mask_child=mask_child, mask_mom=mask_mom
            )
            src_len = batch_src.size(1)
            batch_ground = torch.cat([batch_src[:, src_len // 2:, :], batch_tgt], dim=1)
            #print(batch_ground.shape, pred.shape, batch_src.shape, batch_tgt.shape)
            loss = masked_mse_loss(pred, batch_ground, mask_child=mask_child, mask_mom=mask_mom)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_src.size(0)
        
        avg_train_loss = total_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for val_src, val_tgt in test_loader:
                val_src = val_src.to(device)
                val_tgt = val_tgt.to(device)

                val_child_src = val_src[..., :2]
                val_mom_src = val_src[..., 2:]
                
                val_tgt_in = torch.zeros_like(val_tgt)
                val_tgt_in[:, 1:, :] = val_tgt[:, :-1, :]
                val_child_tgt_in = val_tgt_in[..., :2]
                val_mom_tgt_in = val_tgt_in[..., 2:]

                val_pred = model(
                    val_child_src, val_mom_src,
                    val_child_tgt_in, val_mom_tgt_in,
                    mask_child=False, mask_mom=False,
                )
                val_ground = torch.cat([val_src[:, val_src.size(1) // 2:, :], val_tgt], dim=1)
                # Usually we validate on the full unmasked loss
                val_loss_accum += masked_mse_loss(
                    val_pred, val_ground, mask_child=False, mask_mom=False
                ).item() * val_src.size(0)
                
        avg_val_loss = val_loss_accum / len(test_loader.dataset)

        # --- Logging ---
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.8f} | Val Loss: {avg_val_loss:.8f}")

        # --- Scheduler & Early Stopping Steps ---
        
        # 1. Step Scheduler (reduces LR if validation loss plateaus)
        scheduler.step(avg_val_loss)

        # 2. Check Early Stopping
        early_stopping(avg_val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered. Loading best model weights...")
            model.load_state_dict(torch.load('best_model.pt'))
            break

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict()}, save_path)
        print(f"Saved model to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train masked transformer on month data")
    parser.add_argument("--month", type=int, default=2, help="Month number used in folder pattern diti_*_{month}")
    parser.add_argument("--data-root", type=Path, default=Path("/net/tscratch/people/plgfimpro/korelacje/short_fixed_results_openface2"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input-length", type=int, default=100)
    parser.add_argument("--output-length", type=int, default=25)
    parser.add_argument("--save-path", type=Path, default=Path("masked_model.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_masked_model(
        month=args.month,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        input_length=args.input_length,
        output_length=args.output_length,
        device=device,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
