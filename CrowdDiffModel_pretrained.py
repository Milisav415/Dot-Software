import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sinusoidal timestep embedding (used for conditioning)
def timestep_embedding(timesteps, channels):
    half_dim = channels // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if channels % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

# A basic residual block with optional time-embedding conditioning
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, temb_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels) if temb_channels > 0 else None
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.relu(self.norm1(x)))
        if self.temb_proj is not None:
            h = h + self.temb_proj(F.relu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.relu(self.norm2(h))))
        return h + self.shortcut(x)

# U-Net based diffusion network (downsampling, middle, and upsampling)
class UNetModel(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels,
                 num_res_blocks, attention_resolutions, dropout=0.0,
                 channel_mult=(1, 2, 4, 8), num_head_channels=64,
                 use_scale_shift_norm=False, temb_dim=512):
        super().__init__()
        self.model_channels = model_channels
        self.temb = nn.Sequential(
            nn.Linear(model_channels, temb_dim),
            nn.ReLU(),
            nn.Linear(temb_dim, temb_dim),
        )
        # Input convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        # (Here you would construct the downsampling, middle, and upsampling blocks using ResidualBlock)
        # For brevity, assume 'self.middle' and 'self.output' are defined appropriately.
        self.middle = ResidualBlock(model_channels, model_channels, dropout, temb_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.ReLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, timesteps):
        temb = self.temb(timestep_embedding(timesteps, self.model_channels))
        h = self.input_conv(x)
        # Downsampling and upsampling blocks would process 'h' here...
        h = self.middle(h, temb)
        out = self.output(h)
        return out

# Lightweight counting branch (only used during training)
class CountingDecoder(nn.Module):
    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

# The full CrowdDiff model combining the diffusion UNet and the counting branch
class CrowdDiffModel(nn.Module):
    def __init__(self, in_channels=3, model_channels=192, out_channels=1,
                 num_res_blocks=2, attention_resolutions="32,16,8",
                 dropout=0.0, channel_mult=(1,2,4,8), num_head_channels=64,
                 use_scale_shift_norm=True, temb_dim=512):
        super().__init__()
        self.unet = UNetModel(
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            num_head_channels=num_head_channels,
            use_scale_shift_norm=use_scale_shift_norm,
            temb_dim=temb_dim,
        )
        self.counting_decoder = CountingDecoder(model_channels)

    def forward(self, x, timesteps, return_count=False):
        density_map = self.unet(x, timesteps)
        if return_count:
            count = self.counting_decoder(density_map)
            return density_map, count
        return density_map


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the model with the same parameters used during training.
    model = CrowdDiffModel(
        in_channels=3,
        model_channels=192,
        out_channels=1,
        num_res_blocks=2,
        dropout=0.0,
        temb_dim=512,
    ).to(device)

    # Path to the pre-trained weights (update as necessary)
    pretrained_model_path = r"C:\Users\jm190\Desktop\pre-trained_models\64_256_upsampler.pt"

    try:
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        # If the checkpoint contains a 'state_dict' key, use it.
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print("Pretrained CrowdDiff model loaded successfully!")
    except Exception as e:
        print("Error loading pre-trained model:", e)
        return

    model.eval()

    # Create a dummy input (e.g., a 256x256 RGB image)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    # Use a dummy timestep (choose any value between 0 and the total diffusion steps, e.g., 50)
    timesteps = torch.tensor([50]).to(device)

    with torch.no_grad():
        # The forward pass returns both a density map and a count if return_count=True.
        density_map, count = model(dummy_input, timesteps, return_count=True)

    print("Output density map shape:", density_map.shape)
    print("Predicted crowd count:", count.item())

if __name__ == "__main__":
    main()