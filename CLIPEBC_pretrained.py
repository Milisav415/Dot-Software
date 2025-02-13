import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os


# Example placeholder for the CLIP-EBC model.
# Replace this with the actual architecture of your CLIP-EBC model.
class CLIP_EBC(nn.Module):
    def __init__(self):
        super(CLIP_EBC, self).__init__()
        # Dummy visual prompt tokens (vpt) - the checkpoint contains keys "vpt_0", "vpt_1", etc.
        self.vpt = nn.ParameterList([nn.Parameter(torch.zeros(1, 768)) for _ in range(10)])
        # Dummy logit scale parameter.
        self.logit_scale = nn.Parameter(torch.ones([]))

        # Dummy image encoder: we create a placeholder module with attributes that match the checkpoint.
        self.image_encoder = nn.Module()
        # Create dummy parameters to hold expected keys.
        self.image_encoder.class_embedding = nn.Parameter(torch.zeros(1, 768))
        self.image_encoder.positional_embedding = nn.Parameter(torch.zeros(1, 197, 768))
        self.image_encoder.conv1 = nn.Conv2d(3, 768, kernel_size=16, stride=16, bias=False)
        self.image_encoder.ln_pre = nn.LayerNorm(768)

        # For the transformer blocks, we create a dummy sequential list.
        transformer_blocks = []
        for i in range(12):  # Assume 12 blocks (adjust as necessary)
            block = nn.Module()
            block.attn = nn.Module()
            # Dummy attention parameters
            block.attn.in_proj_weight = nn.Parameter(torch.zeros(768 * 3, 768))
            block.attn.in_proj_bias = nn.Parameter(torch.zeros(768 * 3))
            block.attn.out_proj = nn.Linear(768, 768)
            block.ln_1 = nn.LayerNorm(768)
            # Dummy MLP parameters
            block.mlp = nn.Module()
            block.mlp.c_fc = nn.Linear(768, 3072)
            block.mlp.c_proj = nn.Linear(3072, 768)
            block.ln_2 = nn.LayerNorm(768)
            transformer_blocks.append(block)
        self.image_encoder.transformer = nn.ModuleList(transformer_blocks)

        # Dummy text encoder (to match the keys in the checkpoint)
        self.text_encoder = nn.Module()
        self.text_encoder.positional_embedding = nn.Parameter(torch.zeros(1, 77, 512))
        self.text_encoder.token_embedding = nn.Embedding(49408, 512)
        # We create a dummy transformer for text
        text_transformer_blocks = []
        for i in range(12):  # Assume 12 blocks; adjust as needed.
            block = nn.Module()
            block.attn = nn.Module()
            block.attn.in_proj_weight = nn.Parameter(torch.zeros(512 * 3, 512))
            block.attn.in_proj_bias = nn.Parameter(torch.zeros(512 * 3))
            block.attn.out_proj = nn.Linear(512, 512)
            block.ln_1 = nn.LayerNorm(512)
            block.mlp = nn.Module()
            block.mlp.c_fc = nn.Linear(512, 2048)
            block.mlp.c_proj = nn.Linear(2048, 512)
            block.ln_2 = nn.LayerNorm(512)
            text_transformer_blocks.append(block)
        self.text_encoder.transformer = nn.ModuleList(text_transformer_blocks)
        self.text_encoder.ln_final = nn.LayerNorm(512)
        self.text_encoder.text_projection = nn.Parameter(torch.zeros(512, 512))

        # Dummy image decoder that outputs a heatmap.
        self.image_decoder = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1)
        )
        # Dummy projection layer.
        self.projection = nn.Linear(768, 768)

    def forward(self, x):
        x = self.features(x)
        heatmap = self.heatmap_layer(x)
        return heatmap


def load_model(model_path, device):
    model = CLIP_EBC().to(device)
    try:
        # Load the checkpoint; check if state_dict is a subkey.
        checkpoint = torch.load(model_path, map_location=device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        print("Pretrained CLIP-EBC model loaded successfully!")
    except Exception as e:
        print("Error loading model:", e)
        exit(1)
    model.eval()
    return model


def process_image(img_path, transform):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print("Error loading image:", e)
        exit(1)
    img_tensor = transform(img).unsqueeze(0)  # add batch dimension
    return img, img_tensor


def visualize_heatmap(original_img, heatmap):
    # Normalize the heat map for visualization.
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    plt.figure(figsize=(12, 6))

    # Show the original image.
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")

    # Show the heat map.
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap_norm, cmap='jet')
    plt.colorbar()
    plt.title("Heatmap")
    plt.axis("off")

    plt.show()


def main():
    # Set the device (GPU if available, else CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path to the pre-trained model.
    model_path = r"C:\Users\jm190\Desktop\pre-trained_models\CLIPEBC_best_mae_0.pth"  # Update this with the correct filename/path.

    # Load the pre-trained CLIP-EBC model.
    model = load_model(model_path, device)

    # Define image transformation.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Set the path to your test image.
    img_path = r"C:\Users\jm190\Desktop\jhu_crowd_v2.0\train\images\1213.jpg"  # Replace with your image file path.

    # Process the image.
    original_img, img_tensor = process_image(img_path, transform)
    img_tensor = img_tensor.to(device)

    # Run the model to get the heat map.
    with torch.no_grad():
        heatmap_tensor = model(img_tensor)
        # Remove batch and channel dimensions (assumes output shape [1, 1, H, W]).
        heatmap = heatmap_tensor.squeeze(0).squeeze(0).cpu().numpy()

    # Visualize the original image and the heat map.
    visualize_heatmap(original_img, heatmap)


if __name__ == '__main__':
    main()
