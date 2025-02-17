import matplotlib
matplotlib.use("TkAgg")  # Use the TkAgg backend for popup windows
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Define the CSRNet architecture.
class CSRNetPt(nn.Module):
    def __init__(self):
        super(CSRNetPt, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def segment_image(img, patch_size):
    """
    Segments the input PIL image into non-overlapping patches of size patch_size x patch_size.
    If the image dimensions are not multiples of patch_size, the final segments in each row/column
    will be smaller.
    """
    segments = []
    w, h = img.size
    for top in range(0, h, patch_size):
        for left in range(0, w, patch_size):
            box = (left, top, min(left + patch_size, w), min(top + patch_size, h))
            segment = img.crop(box)
            segments.append((segment, box))
    return segments


def custom_preprocess(img_path):
    # Open the image and convert to RGB
    img = Image.open(img_path).convert('RGB')
    # Convert to a tensor and scale it to the [0,255] range
    img_tensor = 255.0 * TF.to_tensor(img)
    # Subtract the channel-specific means (in pixel space)
    img_tensor[0, :, :] = img_tensor[0, :, :] - 92.8207477031
    img_tensor[1, :, :] = img_tensor[1, :, :] - 95.2757037428
    img_tensor[2, :, :] = img_tensor[2, :, :] - 104.877445883
    return img_tensor

def main():
    # Set the path to your pretrained CSRNet model.
    model_path = "PartAmodel_best.pth"  # Update with the correct filename/path.

    # Create the model instance.
    model = CSRNetPt()
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict)
        print("Pretrained CSRNet model loaded successfully!")
    except Exception as e:
        print("Error loading model:", e)
        return
    model.eval()

    # Define the preprocessing transformation.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Example mean values.
                             std=[0.229, 0.224, 0.225])   # Example std values.
    ])

    # Set the path to your input image.
    img_path = r"C:\Users\jm190\Desktop\jhu_crowd_v2.0\train\images\0283.jpg"  # Replace with your image file path.

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print("Error loading image:", e)
        return

    # Convert the original image to a tensor.
    img_tensor = transform(img)
    # Permute the tensor to H x W x C for visualization.
    img_tensor_np = img_tensor.permute(1, 2, 0).numpy()
    # Unnormalize the image for display.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_unnorm = img_tensor_np
    img_unnorm = np.clip(img_unnorm, 0, 1)

    # Display the image after conversion to a tensor in a popup window.
    plt.figure()
    plt.imshow(img_unnorm)
    plt.title("Image after ToTensor Conversion")
    plt.axis("off")
    plt.show()

    # Segment the original image.
    patch_size = 512  # Adjust patch size as needed.
    segments = segment_image(img, patch_size)
    total_count = 0.0
    density_maps = []
    counts = []

    # Process each segment.
    for idx, (segment, box) in enumerate(segments):
        seg_tensor = transform(segment)
        seg_tensor = seg_tensor.unsqueeze(0)  # Add batch dimension.

        with torch.no_grad():
            density_map = model(seg_tensor)
            density_map = density_map.squeeze(0).squeeze(0)
            count = density_map.sum().item()

        print(f"Segment {idx+1} (box {box}) count: {count:.2f}")
        total_count += count
        density_maps.append(density_map.cpu().numpy())
        counts.append(count)
    print("Total estimated count in the image: {:.2f}".format(total_count))

    density_map_while_img = model(img_tensor)
    print(f'Feeding the whole image we get: {density_map_while_img.sum().item()}')

    # Calculate grid dimensions based on original image dimensions.
    w, h = img.size
    rows = math.ceil(h / patch_size)
    cols = math.ceil(w / patch_size)

    # Create grid plot for heat maps.
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < len(density_maps):
            im = ax.imshow(density_maps[idx], cmap='jet')
            ax.set_title(f"Seg {idx+1}: {counts[idx]:.2f}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()  # This will open the grid plot in a popup window.

if __name__ == '__main__':
    main()
