import torch
import torch.nn as nn
import numpy as np
from utils import *
from models.maskedTimeSeriesTransformerWithHistory import MaskedTimeSeriesTransformerWithHistory as MaskedTimeSeriesTransformer

lr = 1e-3 # Increased slightly as scheduler will handle reduction
epochs = 500 # Increased max epochs since early stopping will catch it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Assuming Model and Loaders are defined elsewhere
model = MaskedTimeSeriesTransformer(d_model=16).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Initialize Scheduler
# Factor 0.5 means LR becomes LR * 0.5
# Patience 3 means wait 3 epochs of no improvement before reducing LR
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
        
        loss = masked_mse_loss(pred, batch_tgt, mask_child=mask_child, mask_mom=mask_mom)
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
            
            # Usually we validate on the full unmasked loss
            val_loss_accum += masked_mse_loss(
                val_pred, val_tgt, mask_child=False, mask_mom=False
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