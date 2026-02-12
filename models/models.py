import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 4,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.output_proj = nn.Linear(d_model, output_dim)
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src = self.input_proj(src)
        src = self.pos_encoder(src)
        tgt = self.input_proj(tgt)
        tgt = self.pos_decoder(tgt)
        out = self.transformer(src, tgt)
        out = self.output_proj(out)
        return out


class MaskedTimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Single projection for concatenated child+mom raw features (4 -> d_model)
        self.input_proj = nn.Linear(4, d_model)

        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.output_proj = nn.Linear(d_model, 4)

        # Missing tokens stay in raw feature space (2 dims each) so concat keeps 4 dims
        self.mom_missing_token = nn.Parameter(torch.zeros(1, 1, 2))
        self.child_missing_token = nn.Parameter(torch.zeros(1, 1, 2))

        nn.init.normal_(self.mom_missing_token, std=0.02)
        nn.init.normal_(self.child_missing_token, std=0.02)

    def forward(
        self,
        child_src: torch.Tensor,
        mom_src: torch.Tensor,
        child_tgt: torch.Tensor,
        mom_tgt: torch.Tensor,
        mask_child: bool = False,
        mask_mom: bool = False,
    ) -> torch.Tensor:
        
        # child_src/mom_src: (B, S, 2), child_tgt/mom_tgt: (B, T, 2)
        batch_size, src_len, _ = child_src.shape

        child_src_emb = child_src
        mom_src_emb = mom_src

        if mask_child:
            child_src_emb = self.child_missing_token.expand(batch_size, src_len, -1)
        if mask_mom:
            mom_src_emb = self.mom_missing_token.expand(batch_size, src_len, -1)

        src_concat = torch.cat([child_src_emb, mom_src_emb], dim=-1)  # (B, S, 4)
        src = self.pos_encoder(self.input_proj(src_concat))

        batch_size, tgt_len, _ = child_tgt.shape
        child_tgt_emb = child_tgt
        mom_tgt_emb = mom_tgt

        if mask_child:
            child_tgt_emb = self.child_missing_token.expand(batch_size, tgt_len, -1)
        if mask_mom:
            mom_tgt_emb = self.mom_missing_token.expand(batch_size, tgt_len, -1)
        tgt_concat = torch.cat([child_tgt_emb, mom_tgt_emb], dim=-1)  # (B, T, 4)
        tgt = self.pos_decoder(self.input_proj(tgt_concat))

        out = self.transformer(src, tgt)
        out = self.output_proj(out)
        return out

    def forward_autoregressive(
        self,
        child_src: torch.Tensor,
        mom_src: torch.Tensor,
        child_tgt: torch.Tensor = None,
        mom_tgt: torch.Tensor = None,
        mask_child: bool = False,
        mask_mom: bool = False,
        tgt_len: int = None,
    ) -> torch.Tensor:
        # child_src/mom_src: (B, S, 2), child_tgt/mom_tgt: (B, T, 2)
        batch_size, src_len, _ = child_src.shape

        child_src_emb = child_src
        mom_src_emb = mom_src

        if mask_child:
            child_src_emb = self.child_missing_token.expand(batch_size, src_len, -1)
        if mask_mom:
            mom_src_emb = self.mom_missing_token.expand(batch_size, src_len, -1)

        src_concat = torch.cat([child_src_emb, mom_src_emb], dim=-1)  # (B, S, 4)
        src = self.pos_encoder(self.input_proj(src_concat))

        # If both targets provided -> teacher-forcing / training mode
        if (child_tgt is not None) and (mom_tgt is not None):
            batch_size, tgt_len, _ = child_tgt.shape
            child_tgt_emb = child_tgt
            mom_tgt_emb = mom_tgt

            if mask_child:
                child_tgt_emb = self.child_missing_token.expand(batch_size, tgt_len, -1)
            if mask_mom:
                mom_tgt_emb = self.mom_missing_token.expand(batch_size, tgt_len, -1)
            tgt_concat = torch.cat([child_tgt_emb, mom_tgt_emb], dim=-1)  # (B, T, 4)
            tgt = self.pos_decoder(self.input_proj(tgt_concat))

            out = self.transformer(src, tgt)
            out = self.output_proj(out)
            return out

        generated = []
        cur = None
        for t in range(tgt_len):
            if cur is None:
                #TODO try with zeros instead of missing tokens
                start_child = self.child_missing_token.expand(batch_size, 1, -1)
                start_mom = self.mom_missing_token.expand(batch_size, 1, -1)
                cur_concat = torch.cat([start_child, start_mom], dim=-1)  # (B,1,4)
            else:
                cur_concat = cur

            # embed current tgt inputs
            tgt_embed = self.pos_decoder(self.input_proj(cur_concat))
            out = self.transformer(src, tgt_embed)  # (B, cur_len, d_model)
            out_raw = self.output_proj(out)  # (B, cur_len, 4)

            last = out_raw[:, -1:, :]
            generated.append(last)

            cur = torch.cat([cur_concat, last.detach()], dim=1)

        gen = torch.cat(generated, dim=1)  # (B, tgt_len, 4)
        return gen
