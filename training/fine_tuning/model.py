import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


class TextEncoder(nn.Module):
    """
    Encodes tokenized prompt text into semantic conditioning representations.
    Provides both:
    1. Full sequence token embeddings (B, L, embed_dim) for multi-token cross-attention.
    2. Global pooled vector (B, embed_dim) for semantic alignment and global conditioning.
    Can utilize local pretrained MiniLM or learned neural embeddings.
    """
    def __init__(self, vocab_size: int = 30522, max_length: int = 128, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embeddings = nn.Parameter(torch.randn(1, max_length, embed_dim) * 0.02)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Optional pretrained semantic backbone
        self._hf_model = None
        self._hf_proj = None
        if self.vocab_size >= 30522:
            self._init_semantic_backbone()

    def _init_semantic_backbone(self):
        try:
            from transformers import AutoModel
            hf = AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2",
                local_files_only=True
            )
            self._hf_model = hf
            self._hf_proj = nn.Linear(384, self.embed_dim)
        except Exception:
            self._hf_model = None
            self._hf_proj = None

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        token_ids: (B, L)
        Returns:
          token_context: (B, L, embed_dim)
          pooled: (B, embed_dim)
        """
        b, seq_len = token_ids.shape
        device = token_ids.device

        # If semantic backbone is loaded and token_ids match BERT/MiniLM vocab
        if self._hf_model is not None and token_ids.max() < 30522:
            try:
                attention_mask = (token_ids != 0).long().to(device)
                with torch.no_grad():
                    hf_out = self._hf_model(input_ids=token_ids, attention_mask=attention_mask)
                # hf_out.last_hidden_state: (B, L, 384)
                token_context = self._hf_proj(hf_out.last_hidden_state.to(device))
                mask_f = attention_mask.unsqueeze(-1).float()
                pooled = (token_context * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
                return token_context, pooled
            except Exception:
                pass

        # Standard learned embedding path
        safe_tokens = token_ids.clamp(0, self.token_embeddings.num_embeddings - 1)
        max_pos = self.pos_embeddings.shape[1]
        if seq_len > max_pos:
            safe_tokens = safe_tokens[:, :max_pos]
            seq_len = max_pos
            
        pos = self.pos_embeddings[:, :seq_len, :].to(device)
        x = self.token_embeddings(safe_tokens) + pos
        x = self.norm(x)
        token_context = self.proj(x) # (B, seq_len, embed_dim)
        
        mask = (safe_tokens != 0).unsqueeze(-1).float()
        pooled = (token_context * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return token_context, pooled


class TemporalResidualBlock(nn.Module):
    """
    Temporal residual convolution and multi-token cross-attention block.
    Conditions spatial-temporal features on full prompt token sequences.
    """
    def __init__(self, channels: int = 64, text_dim: int = 128):
        super().__init__()
        self.channels = channels
        self.norm1 = nn.GroupNorm(8, channels)
        # 1D temporal convolution across time dimension
        self.temporal_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(8, channels)
        # Multi-token cross-attention projections
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(text_dim, channels)
        self.to_v = nn.Linear(text_dim, channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, text_context: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C, H, W)
        text_context: (B, L, text_dim) or (B, text_dim)
        """
        b, t, c, h, w = x.shape

        # 1. Temporal 1D convolution
        # Reshape to (B * H * W, C, T)
        x_perm = x.permute(0, 3, 4, 2, 1).contiguous().view(b * h * w, c, t)
        h_temp = F.relu(self.temporal_conv(x_perm))
        # Reshape back to (B, T, C, H, W)
        h_temp = h_temp.view(b, h, w, c, t).permute(0, 4, 3, 1, 2)
        x = x + h_temp

        # 2. Text cross-attention
        # Query from spatial-temporal features: (B, T * H * W, C)
        x_flat = x.permute(0, 1, 3, 4, 2).contiguous().view(b, t * h * w, c)
        q = self.to_q(x_flat) # (B, t*h*w, C)

        # Ensure text_context is 3D: (B, L, text_dim)
        if text_context.dim() == 2:
            # (B, text_dim) -> (B, 1, text_dim)
            text_seq = text_context.unsqueeze(1)
        else:
            text_seq = text_context # (B, L, text_dim)

        k = self.to_k(text_seq) # (B, L, C)
        v = self.to_v(text_seq) # (B, L, C)

        # Compute real multi-token cross-attention scores across sequence length L
        # (B, t*h*w, C) @ (B, C, L) -> (B, t*h*w, L)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(c)
        attn = F.softmax(scores, dim=-1) # Softmax over sequence tokens L

        # Attend to token values
        # (B, t*h*w, L) @ (B, L, C) -> (B, t*h*w, C)
        out = torch.bmm(attn, v)
        out = self.proj(out).view(b, t, h, w, c).permute(0, 1, 4, 2, 3)

        return x + out


class SpatialTemporalTTVModel(nn.Module):
    """
    Genuine PyTorch Spatial-Temporal Neural Text-to-Video Architecture.
    Consists of:
    - Text Conditioning Encoder (multi-token sequence + pooled embedding)
    - Spatial Keyframe Encoder (downsamples 4x)
    - Temporal Position Embedding
    - Temporal Motion Synthesizer Blocks with Multi-Token Cross-Attention
    - Spatial Frame Decoder (upsamples 4x back to image space)
    - Semantic Video-Text Alignment Projection Head
    """
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 64,
        num_frames: int = 8,
        vocab_size: int = 30522,
        max_prompt_length: int = 128,
        text_embed_dim: int = 128
    ):
        super().__init__()
        self.num_frames = num_frames
        self.latent_channels = latent_channels
        self.text_embed_dim = text_embed_dim
        self.max_prompt_length = max_prompt_length

        # 1. Text conditioning
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            max_length=max_prompt_length,
            embed_dim=text_embed_dim
        )

        # 2. Spatial Keyframe Encoder (downsamples 4x)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), # /2
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, latent_channels, kernel_size=3, stride=2, padding=1), # /4
            nn.GroupNorm(8, latent_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 3. Temporal Position Embedding
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, num_frames, latent_channels, 1, 1) * 0.02)

        # 4. Temporal Motion Synthesizer Blocks
        self.temp_block1 = TemporalResidualBlock(latent_channels, text_embed_dim)
        self.temp_block2 = TemporalResidualBlock(latent_channels, text_embed_dim)

        # 5. Spatial Frame Decoder (upsamples 4x back to image space)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 32, kernel_size=4, stride=2, padding=1), # x2
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1), # x4
            nn.Tanh() # Normalizes output to [-1.0, 1.0]
        )

        # 6. Alignment projection head for semantic similarity
        self.video_semantic_proj = nn.Linear(latent_channels, text_embed_dim)

    def forward(
        self,
        keyframe: torch.Tensor,
        prompt_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Inputs:
          keyframe: Initial frame tensor (B, C, H, W)
          prompt_tokens: Tokenized prompt IDs (B, L)
        Returns:
          generated_frames: (B, T, C, H, W) normalized to [-1.0, 1.0]
          video_embedding: (B, text_embed_dim) for semantic alignment loss
          text_embedding: (B, text_embed_dim)
        """
        b = keyframe.shape[0]

        # 1. Encode text context: token_seq is (B, L, D), pooled is (B, D)
        token_seq, text_pooled = self.text_encoder(prompt_tokens)

        # 2. Encode initial keyframe
        z0 = self.spatial_encoder(keyframe) # (B, latent_channels, H/4, W/4)

        # 3. Expand across time dimension
        z_seq = z0.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1) # (B, T, C, h, w)
        t_avail = min(self.num_frames, self.temporal_pos_embed.shape[1])
        z_seq[:, :t_avail] = z_seq[:, :t_avail] + self.temporal_pos_embed[:, :t_avail, :, :, :]

        # 4. Synthesize motion conditioned on multi-token sequence
        z_seq = self.temp_block1(z_seq, token_seq)
        z_seq = self.temp_block2(z_seq, token_seq)

        # 5. Extract global video representation for semantic alignment
        video_emb = self.video_semantic_proj(z_seq.mean(dim=(1, 3, 4)))

        # 6. Decode each frame
        b, t, c, h, w = z_seq.shape
        z_flat = z_seq.view(b * t, c, h, w)
        out_flat = self.decoder(z_flat) # (B*T, 3, H, W)
        generated_frames = out_flat.view(b, t, 3, out_flat.shape[2], out_flat.shape[3])

        return generated_frames, video_emb, text_pooled
