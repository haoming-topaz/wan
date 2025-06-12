import torch
import torch.nn as nn

class CoordinateConditionEncoder(nn.Module):
    def __init__(self, num_channels=20, num_abs_freqs=4):
        """
        num_channels: total output channels (C)
        num_abs_freqs: number of frequencies used in sine/cosine encoding (per coordinate)
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_abs_freqs = num_abs_freqs
        
        # Basic sanity check
        min_required_channels = 4 + 4 * num_abs_freqs  # delta_x, delta_y, distance, log_distance + abs pos
        if num_channels < min_required_channels:
            raise ValueError(f"num_channels must be at least {min_required_channels} to fit hybrid encoding.")
        
        # Conv1x1 projection layer to normalize and fuse channels
        self.proj = nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=True)

    def forward(self, x, y, H, W):
        """
        x, y: target coordinate, float or tensor of shape (N,) if batching
        H, W: target image size
        Output: (C, H, W) tensor
        """
        device = x.device
        
        # Prepare pixel grid (H, W)
        yy, xx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        yy = yy.to(torch.float32)
        xx = xx.to(torch.float32)        
        
        # delta_x / delta_y with normalization to [-1, 1]
        delta_x_norm = 2.0 * (xx - x) / max(W - 1, 1)
        delta_y_norm = 2.0 * (yy - y) / max(H - 1, 1)
        
        euclidean_dist = torch.sqrt((xx - x) ** 2 + (yy - y) ** 2 + 1e-6)
        log_dist = torch.log1p(euclidean_dist)
        
        # (1, H, W) maps
        delta_x_map = delta_x_norm.unsqueeze(0)
        delta_y_map = delta_y_norm.unsqueeze(0)
        euclidean_dist_map = euclidean_dist.unsqueeze(0)
        log_dist_map = log_dist.unsqueeze(0)
        
        # Global absolute sine/cosine encoding
        def sincos_encoding(v, num_freqs):
            dim_t = 10000 ** (2 * (torch.arange(num_freqs, device=device).float() // 2) / num_freqs)
            v_proj = v / dim_t
            v_enc = torch.stack([v_proj.sin(), v_proj.cos()], dim=-1).flatten()
            return v_enc  # shape (2 * num_freqs,)
        
        abs_x_enc = sincos_encoding(x, self.num_abs_freqs)
        abs_y_enc = sincos_encoding(y, self.num_abs_freqs)
        abs_enc = torch.cat([abs_x_enc, abs_y_enc], dim=0)  # (4 * num_freqs,)
        
        # Broadcast global abs pos encoding
        abs_enc_map = abs_enc[:, None, None].repeat(1, H, W)  # (4 * num_freqs, H, W)
        
        # Compose feature map
        feature_maps = [
            delta_x_map,
            delta_y_map,
            euclidean_dist_map,
            log_dist_map,
            abs_enc_map
        ]
        
        full_feature_map = torch.cat(feature_maps, dim=0)  # (C_current, H, W)
        
        # Pad or truncate to target num_channels
        if full_feature_map.shape[0] < self.num_channels:
            pad_channels = self.num_channels - full_feature_map.shape[0]
            padding = torch.zeros((pad_channels, H, W), device=device, dtype=full_feature_map.dtype)
            full_feature_map = torch.cat([full_feature_map, padding], dim=0)
        elif full_feature_map.shape[0] > self.num_channels:
            full_feature_map = full_feature_map[:self.num_channels]
        
        # Apply Conv1x1 projection (learnable normalization + fusion)
        full_feature_map = self.proj(full_feature_map.unsqueeze(0)).squeeze(0)  # keep (C, H, W)
        
        return full_feature_map
