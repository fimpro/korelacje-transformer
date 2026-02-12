from models.revin import RevIN
import torch
import torch.nn as nn
from models.positionalEncoding import PositionalEncoding

class RevinTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 16,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        # Ensure d_model is divisible by 2 for the split concatenation
        assert d_model % 2 == 0, "d_model must be divisible by 2"
        self.d_half = d_model // 2

        # 1. Projections: Raw (2) -> Embedding (d_model/2)
        self.input_proj_mom = nn.Linear(2, self.d_half)
        self.input_proj_child = nn.Linear(2, self.d_half)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        # 3. Transformer Core
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # 4. Output Projections: d_model -> Raw (4)
        # We project back to 4 coordinates (Child X,Y + Mom X,Y)
        self.output_proj = nn.Linear(d_model, 4)

        # 5. Missing Tokens (Learnable parameters)
        self.mom_missing_token = nn.Parameter(torch.zeros(1, 1, self.d_half))
        self.child_missing_token = nn.Parameter(torch.zeros(1, 1, self.d_half))

        nn.init.normal_(self.mom_missing_token, std=0.02)
        nn.init.normal_(self.child_missing_token, std=0.02)

        self.revin = RevIN(num_features=4, affine=True)

    def _embed_and_mask(self, child, mom, mask_child, mask_mom):
        """Helper to project and optionally mask inputs."""
        B, S, _ = child.shape
        
        # Project
        child_emb = self.input_proj_child(child) # (B, S, d_half)
        mom_emb = self.input_proj_mom(mom)       # (B, S, d_half)

        # Apply Masks (Replace embedding with missing token)
        if mask_child:
            # Expand token to (B, S, d_half)
            token = self.child_missing_token.expand(B, S, -1)
            child_emb = token
        
        if mask_mom:
            token = self.mom_missing_token.expand(B, S, -1)
            mom_emb = token

        # Concatenate to create full d_model vector
        # (B, S, d_half) + (B, S, d_half) -> (B, S, d_model)
        return torch.cat([child_emb, mom_emb], dim=-1)

    def forward(
        self,
        child_src: torch.Tensor,
        mom_src: torch.Tensor,
        child_tgt: torch.Tensor,
        mom_tgt: torch.Tensor,
        mask_child: bool = False,
        mask_mom: bool = False,
        src_key_padding_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Training Forward Pass.
        src args: (B, Src_Len, 2)
        tgt args: (B, Tgt_Len, 2) - This represents the FUTURE we want to predict.
        """
        # If we're in evaluation/inference mode, delegate to `predict()` which
        # runs autoregressive generation under `eval()` + `no_grad()`.
        if not self.training:
            steps = child_tgt.shape[1]
            return self.predict(
                child_src,
                mom_src,
                steps=steps,
                mask_child=mask_child,
                mask_mom=mask_mom,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
        # --- 1. Prepare Encoder Input (Source) ---
        # Keep encoder embeddings in `enc` so we don't overwrite them later.
        enc_raw = torch.cat([child_src, mom_src], dim=-1)  # (B, src_len, 4)
        enc_norm = self.revin(enc_raw, mode='norm')
        enc_child, enc_mom = enc_norm[..., :2], enc_norm[..., 2:]
        enc = self._embed_and_mask(enc_child, enc_mom, mask_child, mask_mom)
        enc = self.pos_encoder(enc)

        # --- 2. Prepare Decoder Input (Target) ---
        # Strategy: Decoder sees last 50% of History + The Future Target
        B, src_len, _ = child_src.shape
        overlap_idx = src_len // 2

        # Extract overlap from source
        child_overlap = child_src[:, overlap_idx:, :]
        mom_overlap = mom_src[:, overlap_idx:, :]
        
        # Concatenate Overlap + Future Target
        # Note: For Teacher Forcing, 'child_tgt' usually contains the Ground Truth 
        decoder_child_in = torch.cat([child_overlap, child_tgt], dim=1)
        decoder_mom_in = torch.cat([mom_overlap, mom_tgt], dim=1)

        # Embed Decoder Input
        # Apply RevIN to raw 4-channel decoder inputs, then embed.
        dec_raw = torch.cat([decoder_child_in, decoder_mom_in], dim=-1)  # (B, dec_len, 4)
        dec_norm = self.revin(dec_raw, mode='norm')
        dec_child, dec_mom = dec_norm[..., :2], dec_norm[..., 2:]
        tgt = self._embed_and_mask(dec_child, dec_mom, mask_child=mask_child, mask_mom=mask_mom)
        tgt = self.pos_decoder(tgt)

        # --- 3. Transformer Masks ---
        T = tgt.shape[1]
        tgt_mask = self.transformer.generate_square_subsequent_mask(T).to(tgt.device)

        # Build decoder key padding mask that accounts for the overlap prefix.
        # The provided `tgt_key_padding_mask` usually refers only to the future target
        # (length = child_tgt.shape[1]). We must prepend `False` for the overlap
        # history positions so the mask length matches `tgt.shape[1]`.
        if tgt_key_padding_mask is not None:
            # tgt_key_padding_mask: (B, target_len)
            B_mask, orig_tgt_len = tgt_key_padding_mask.shape
            overlap_len = tgt.shape[1] - orig_tgt_len
            if overlap_len > 0:
                prefix = torch.zeros((B_mask, overlap_len), dtype=torch.bool, device=tgt_key_padding_mask.device)
                dec_tgt_key_padding_mask = torch.cat([prefix, tgt_key_padding_mask.to(tgt.device)], dim=1)
            else:
                dec_tgt_key_padding_mask = tgt_key_padding_mask.to(tgt.device)
        else:
            dec_tgt_key_padding_mask = None

        # --- 4. Pass through Transformer ---
        out = self.transformer(
            enc,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=dec_tgt_key_padding_mask,
        )
        
        # --- 5. Project to Output ---
        # Project full decoder output to 4 features and denormalize.
        out_proj = self.output_proj(out)  # (B, dec_len, 4)
        out_denorm = self.revin(out_proj, mode='denorm')

        # Return only the predicted future (the last `child_tgt` timesteps)
        targ_len = child_tgt.shape[1]
        return out_denorm[:, -targ_len:, :]

    def predict(self, child_src, mom_src, steps=25, mask_child=False, mask_mom=False, src_key_padding_mask: torch.Tensor = None, tgt_key_padding_mask: torch.Tensor = None):
        """
        Autoregressive Inference.
        """
        self.eval()
        device = child_src.device
        
        with torch.no_grad():
            # 1. Encode the Source History once
            # Apply RevIN to raw 4-channel source before embedding; keep in `enc`.
            enc_raw = torch.cat([child_src, mom_src], dim=-1)
            enc_norm = self.revin(enc_raw, mode='norm')
            enc_child, enc_mom = enc_norm[..., :2], enc_norm[..., 2:]
            enc = self._embed_and_mask(enc_child, enc_mom, mask_child, mask_mom)
            enc = self.pos_encoder(enc)
            
            # 2. Initialize Decoder Input
            # We start with the second half of the history (warmup)
            src_len = child_src.shape[1]
            overlap_idx = src_len // 2
            
            curr_child = child_src[:, overlap_idx:, :]
            curr_mom = mom_src[:, overlap_idx:, :]

            # Do not replace raw mom/child with embedding tokens here; pass mask flags
            # to _embed_and_mask so masking is applied inside embedding step.
            
            # 3. Autoregressive Loop
            for _ in range(steps):
                # Prepare decoder raw inputs (history overlap + generated tokens)
                dec_raw = torch.cat([curr_child, curr_mom], dim=-1)
                dec_norm = self.revin(dec_raw, mode='norm')
                dec_child, dec_mom = dec_norm[..., :2], dec_norm[..., 2:]
                tgt = self._embed_and_mask(dec_child, dec_mom, mask_child=mask_child, mask_mom=mask_mom)
                tgt = self.pos_decoder(tgt)
                
                # Create Causal Mask
                T = tgt.shape[1]
                tgt_mask = self.transformer.generate_square_subsequent_mask(T).to(device)
                
                # Build current tgt_key_padding_mask for growing decoder (if provided)
                if tgt_key_padding_mask is not None:
                    # tgt_key_padding_mask refers to original target length; extend with False for generated tokens
                    cur_len = tgt.shape[1]
                    orig_len = tgt_key_padding_mask.shape[1]
                    if cur_len <= orig_len:
                        cur_tgt_pad = tgt_key_padding_mask[:, :cur_len].to(device)
                    else:
                        extra = cur_len - orig_len
                        extra_pad = torch.zeros((tgt_key_padding_mask.size(0), extra), dtype=torch.bool, device=device)
                        cur_tgt_pad = torch.cat([tgt_key_padding_mask.to(device), extra_pad], dim=1)
                else:
                    cur_tgt_pad = None

                # Transformer Forward
                # We only need the output for the LAST timestep
                out = self.transformer(
                    enc,
                    tgt,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=cur_tgt_pad,
                )
                
                # Project back to 4 coords (Child X,Y, Mom X,Y)
                next_val = self.output_proj(out[:, -1:, :]) # (B, 1, 4)
                next_val = self.revin(next_val, mode='denorm')
                
                # Split result
                next_child = next_val[..., :2]
                next_mom = next_val[..., 2:]
                
                # Append prediction to decoder input for next iteration
                curr_child = torch.cat([curr_child, next_child], dim=1)
                curr_mom = torch.cat([curr_mom, next_mom], dim=1)

            # Return only the predicted future (removing the overlap history)
            # curr_child contains [Overlap (50) + Prediction (25)]
            pred_child = curr_child[:, -steps:, :]
            pred_mom = curr_mom[:, -steps:, :]
            
            return torch.cat([pred_child, pred_mom], dim=-1)
    def forward_autoregressive(
        self,
        child_src: torch.Tensor,
        mom_src: torch.Tensor,
        mask_child: bool = False,
        mask_mom: bool = False,
        tgt_len: int = None,
        src_key_padding_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Performs autoregressive decoding sequence generation.
        Matches signature of forward(), ignoring actual tgt values (uses length only).
        Detailed logic mirrors predict() but without forcing eval/no_grad context.
        """
        device = child_src.device
        steps = tgt_len if tgt_len is not None else 25

        # 1. Encode Source
        # Apply RevIN to raw 4-channel source before embedding and keep in `enc`.
        src_raw = torch.cat([child_src, mom_src], dim=-1)
        src_norm = self.revin(src_raw, mode='norm')
        src_child, src_mom = src_norm[..., :2], src_norm[..., 2:]
        enc = self._embed_and_mask(src_child, src_mom, mask_child=mask_child, mask_mom=mask_mom)
        enc = self.revin(enc, mode='norm') if False else enc
        enc = self.pos_encoder(enc)

        # 2. Initialize Decoder Input with History Overlap
        src_len = child_src.shape[1]
        overlap_idx = src_len // 2
        curr_child = child_src[:, overlap_idx:, :]
        curr_mom = mom_src[:, overlap_idx:, :]

        # 3. Step-by-step Generation (preserve gradients)
        for _ in range(steps):
            dec_raw = torch.cat([curr_child, curr_mom], dim=-1)
            dec_norm = self.revin(dec_raw, mode='norm')
            dec_child, dec_mom = dec_norm[..., :2], dec_norm[..., 2:]
            tgt = self._embed_and_mask(dec_child, dec_mom, mask_child=mask_child, mask_mom=mask_mom)
            tgt = self.pos_decoder(tgt)

            seq_len = tgt.shape[1]
            tgt_mask = self.transformer.generate_square_subsequent_mask(seq_len).to(device)

            # build current tgt padding mask if provided
            if tgt_key_padding_mask is not None:
                orig_len = tgt_key_padding_mask.shape[1]
                if seq_len <= orig_len:
                    cur_tgt_pad = tgt_key_padding_mask[:, :seq_len].to(device)
                else:
                    extra = seq_len - orig_len
                    extra_pad = torch.zeros((tgt_key_padding_mask.size(0), extra), dtype=torch.bool, device=device)
                    cur_tgt_pad = torch.cat([tgt_key_padding_mask.to(device), extra_pad], dim=1)
            else:
                cur_tgt_pad = None

            out = self.transformer(
                enc,
                tgt,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=cur_tgt_pad,
            )

            next_token = self.output_proj(out[:, -1:, :])
            next_token = self.revin(next_token, mode='denorm')

            next_child = next_token[..., :2]
            next_mom = next_token[..., 2:]

            curr_child = torch.cat([curr_child, next_child], dim=1)
            curr_mom = torch.cat([curr_mom, next_mom], dim=1)

        pred_child = curr_child[:, -steps:, :]
        pred_mom = curr_mom[:, -steps:, :]
        return torch.cat([pred_child, pred_mom], dim=-1)
