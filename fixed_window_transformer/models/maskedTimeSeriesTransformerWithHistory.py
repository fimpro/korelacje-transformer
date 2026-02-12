import torch
import torch.nn as nn
from models.positionalEncoding import PositionalEncoding
from models.revin import RevIN

class MaskedTimeSeriesTransformerWithHistory(nn.Module):
    def __init__(
        self,
        d_model: int = 16,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        use_revin: bool = True,
        revin_affine: bool = True,
        revin_eps: float = 1e-5,
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

        # Optional RevIN normalization (on raw 4-channel inputs)
        self.use_revin = use_revin
        self.revin = RevIN(num_features=4, eps=revin_eps, affine=revin_affine) if use_revin else None

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
    ) -> torch.Tensor:
        """
        Training Forward Pass.
        src args: (B, Src_Len, 2)
        tgt args: (B, Tgt_Len, 2) - This represents the FUTURE we want to predict.
        """
        
        # --- 1. Prepare Encoder Input (Source) ---
        if self.use_revin:
            enc_raw = torch.cat([child_src, mom_src], dim=-1)
            enc_norm = self.revin(enc_raw, mode='norm')
            enc_child, enc_mom = enc_norm[..., :2], enc_norm[..., 2:]
            src = self._embed_and_mask(enc_child, enc_mom, mask_child, mask_mom)
        else:
            src = self._embed_and_mask(child_src, mom_src, mask_child, mask_mom)
        src = self.pos_encoder(src)

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
        # Note: We usually DO NOT mask the decoder input during training if we want 
        # the model to learn the relationship. If you strictly want to test imputation,
        # you can pass the masks here too. For now, we assume decoder sees reality.
        if self.use_revin:
            dec_raw = torch.cat([decoder_child_in, decoder_mom_in], dim=-1)
            dec_norm = self.revin(dec_raw, mode='norm')
            dec_child, dec_mom = dec_norm[..., :2], dec_norm[..., 2:]
            tgt = self._embed_and_mask(dec_child, dec_mom, mask_child=mask_child, mask_mom=mask_mom)
        else:
            tgt = self._embed_and_mask(decoder_child_in, decoder_mom_in, mask_child=mask_child, mask_mom=mask_mom)
        tgt = self.pos_decoder(tgt)

        # --- 3. Transformer Masks ---
        T = tgt.shape[1]
        tgt_mask = self.transformer.generate_square_subsequent_mask(T).to(tgt.device)

        # --- 4. Pass through Transformer ---
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        
        # --- 5. Project to Output ---
        out_proj = self.output_proj(out)
        if self.use_revin:
            out_proj = self.revin(out_proj, mode='denorm')
        return out_proj

    def predict(self, child_src, mom_src, steps=25, mask_child=False, mask_mom=False):
        """
        Autoregressive Inference.
        """
        self.eval()
        device = child_src.device
        
        with torch.no_grad():
            # 1. Encode the Source History once
            # We apply masks here if we want to simulate "Mom is missing from history"
            if self.use_revin:
                enc_raw = torch.cat([child_src, mom_src], dim=-1)
                enc_norm = self.revin(enc_raw, mode='norm')
                enc_child, enc_mom = enc_norm[..., :2], enc_norm[..., 2:]
                src = self._embed_and_mask(enc_child, enc_mom, mask_child, mask_mom)
            else:
                src = self._embed_and_mask(child_src, mom_src, mask_child, mask_mom)
            src = self.pos_encoder(src)
            
            # 2. Initialize Decoder Input
            # We start with the second half of the history (warmup)
            src_len = child_src.shape[1]
            overlap_idx = src_len // 2
            
            curr_child = child_src[:, overlap_idx:, :]
            curr_mom = mom_src[:, overlap_idx:, :]

            # 3. Autoregressive Loop
            for _ in range(steps):
                # Embed current decoder input
                if self.use_revin:
                    dec_raw = torch.cat([curr_child, curr_mom], dim=-1)
                    dec_norm = self.revin(dec_raw, mode='norm')
                    dec_child, dec_mom = dec_norm[..., :2], dec_norm[..., 2:]
                    tgt = self._embed_and_mask(dec_child, dec_mom, mask_child=mask_child, mask_mom=mask_mom)
                else:
                    tgt = self._embed_and_mask(curr_child, curr_mom, mask_child=mask_child, mask_mom=mask_mom)
                tgt = self.pos_decoder(tgt)
                
                # Create Causal Mask
                T = tgt.shape[1]
                tgt_mask = self.transformer.generate_square_subsequent_mask(T).to(device)
                
                # Transformer Forward
                # We only need the output for the LAST timestep
                out = self.transformer(src, tgt, tgt_mask=tgt_mask)
                
                # Project back to 4 coords (Child X,Y, Mom X,Y)
                next_val = self.output_proj(out[:, -1:, :]) # (B, 1, 4)
                if self.use_revin:
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
    ) -> torch.Tensor:
        """
        Performs autoregressive decoding sequence generation.
        Matches signature of forward(), ignoring actual tgt values (uses length only).
        Detailed logic mirrors predict() but without forcing eval/no_grad context.
        """
        device = child_src.device
        steps = tgt_len

        # 1. Encode Source
        if self.use_revin:
            enc_raw = torch.cat([child_src, mom_src], dim=-1)
            enc_norm = self.revin(enc_raw, mode='norm')
            enc_child, enc_mom = enc_norm[..., :2], enc_norm[..., 2:]
            src = self._embed_and_mask(enc_child, enc_mom, mask_child=mask_child, mask_mom=mask_mom)
        else:
            src = self._embed_and_mask(child_src, mom_src, mask_child=mask_child, mask_mom=mask_mom)
        src = self.pos_encoder(src)

        # 2. Initialize Decoder Input with History Overlap
        src_len = child_src.shape[1]
        overlap_idx = src_len // 2
        
        curr_child = child_src[:, overlap_idx:, :]
        curr_mom = mom_src[:, overlap_idx:, :]

        # 3. Step-by-step Generation
        for _ in range(steps):
            # Embed
            if self.use_revin:
                dec_raw = torch.cat([curr_child, curr_mom], dim=-1)
                dec_norm = self.revin(dec_raw, mode='norm')
                dec_child, dec_mom = dec_norm[..., :2], dec_norm[..., 2:]
                tgt = self._embed_and_mask(dec_child, dec_mom, mask_child=mask_child, mask_mom=mask_mom)
            else:
                tgt = self._embed_and_mask(curr_child, curr_mom, mask_child=mask_child, mask_mom=mask_mom)
            tgt = self.pos_decoder(tgt)
            
            # Mask
            seq_len = tgt.shape[1]
            tgt_mask = self.transformer.generate_square_subsequent_mask(seq_len).to(device)

            # Transformer Pass
            out = self.transformer(src, tgt, tgt_mask=tgt_mask)
            
            # Predict Next Coordinate
            # (B, T, d_model) -> (B, 1, 4)
            # We only care about the last token's output
            next_token = self.output_proj(out[:, -1:, :])
            if self.use_revin:
                next_token = self.revin(next_token, mode='denorm')
            
            next_child = next_token[..., :2]
            next_mom = next_token[..., 2:]
            
            # Append to input for next iteration
            curr_child = torch.cat([curr_child, next_child], dim=1)
            curr_mom = torch.cat([curr_mom, next_mom], dim=1)

        # 4. Extract Prediction (Remove overlap)
        pred_child = curr_child[:, -steps:, :]
        pred_mom = curr_mom[:, -steps:, :]

        return torch.cat([pred_child, pred_mom], dim=-1)
# --- Standard Positional Encoding (Helper) ---
