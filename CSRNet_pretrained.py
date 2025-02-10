import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# Define the CSRNet architecture.
# IMPORTANT: This architecture must match the one used when training the pretrained model.
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

def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def resize_img(img):
    # Optionally, resize the image while preserving aspect ratio.
    short_side = 1024
    w, h = img.size
    scale = short_side / min(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h))

    # Display the resized image.
    plt.figure()
    plt.imshow(img_resized)
    plt.title("Resized Image")
    plt.axis("off")
    plt.show()

    return img_resized

def main():
    # Set the path to your pretrained CSRNet model.
    model_path = "PartAmodel_best.pth"  # Update this with the correct filename/path.

    # Create the model instance.
    model = CSRNetPt()
    try:
        # Load the checkpoint and extract the state dictionary.
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
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
                             std=[0.229, 0.224, 0.225])  # Example std values.
    ])

    # Set the path to your input image.
    img_path = r"C:\Users\jm190\Desktop\jhu_crowd_v2.0\train\images\1213.jpg"  # Replace with your image file path.
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print("Error loading image:", e)
        return

    img_resized = resize_img(img) # resize the image

    # Preprocess the image.
    img_tensor = transform(img)

    # To display the tensor as an image, convert it back to a NumPy array.
    # The tensor shape is [C, H, W] so we need to permute it to [H, W, C].
    img_tensor_np = img_tensor.permute(1, 2, 0).numpy()

    # Display the image from the tensor.
    plt.figure()
    plt.imshow(img_tensor_np)
    plt.title("Image after ToTensor Transformation")
    plt.axis("off")
    plt.show()

    # Add batch dimension.
    img_tensor = img_tensor.unsqueeze(0)

    # Perform a forward pass through the network to get the density map.
    with torch.no_grad():
        density_map = model(img_tensor)
        # Remove batch and channel dimensions: expected shape becomes [H, W].
        density_map = density_map.squeeze(0).squeeze(0)

    # Compute the estimated count by summing over the density map.
    estimated_count = density_map.sum().item()
    print("Estimated count: {:.2f}".format(estimated_count))

    # Visualize the density map.
    density_map_np = density_map.cpu().numpy()
    plt.imshow(density_map_np, cmap='jet')
    plt.colorbar()
    plt.title("Density Map")
    plt.show()


if __name__ == '__main__':
    main()
