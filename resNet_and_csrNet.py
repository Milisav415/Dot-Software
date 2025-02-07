import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import os


# Improved Dataset Class with error checking
class CountDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Initializes the CountDataset.

        Parameters:
            root_dir (str): Root directory of the dataset.
            split (str): Sub-directory split, e.g., 'train' or 'val'.
            transform (callable, optional): Transformations to apply to each image.
        """
        # Set up directory paths based on the provided root directory and split.
        self.split_dir = os.path.join(root_dir, split)
        self.images_dir = os.path.join(self.split_dir, "images")
        self.labels_file = os.path.join(self.split_dir, "image_labels.txt")
        self.transform = transform

        # Validate that the images directory exists.
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Image directory {self.images_dir} not found!")
        # Validate that the labels file exists.
        if not os.path.exists(self.labels_file):
            raise FileNotFoundError(f"Label file {self.labels_file} not found!")

        # Parse the labels file to create a dictionary mapping image IDs to counts.
        self.labels = self._parse_labels()
        # Validate image file paths and store only the valid ones.
        self.image_files = self._validate_image_paths()

    def _parse_labels(self):
        """
        Parses the labels file and returns a dictionary mapping image IDs to counts.

        Returns:
            dict: Dictionary where keys are image IDs and values are counts.
        """
        labels = {}
        try:
            with open(self.labels_file, 'r') as f:
                for line in f:
                    # Split the line by commas (e.g., "img_id,count,other")
                    parts = line.strip().split(',') # split by coma
                    if len(parts) >= 2:
                        img_id = parts[0]
                        try:
                            # Convert the count to an integer
                            count = int(parts[1])
                        except ValueError:
                            print(f"Could not parse {parts[1]} as an int for image {img_id}")
                        # Ensure that the count is non-negative
                        labels[img_id] = max(0.0, count)  # Ensure non-negative counts
        except Exception as e:
            raise RuntimeError(f"Error parsing labels: {str(e)}")
        return labels

    def _validate_image_paths(self):
        """
        Validates the existence of image files for each label entry.

        Returns:
            list: List of valid image file paths.
        """
        valid_files = []
        for img_id in self.labels.keys():
            # Check for different common image extensions.
            for ext in ['.jpg', '.png', '.jpeg']:  # Support multiple extensions
                img_path = os.path.join(self.images_dir, f"{img_id}{ext}")
                if os.path.exists(img_path):
                    valid_files.append(img_path)
                    break # Stop after finding the first matching file.
            else:
                # Warn if no image file is found for the given image ID.
                print(f"Warning: No image found for {img_id}")
        return valid_files

    def __len__(self):
        """
        Returns the total number of valid image files.
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Retrieves the image and its corresponding count for a given index.

        Parameters:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image tensor, count tensor)
        """
        # Get the image path based on the index.
        img_path = self.image_files[idx]
        # Extract the image ID (filename without extension).
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        # Open the image and convert it to RGB format.
        image = Image.open(img_path).convert("RGB")
        # Apply transformations if provided.
        if self.transform:
            image = self.transform(image)
        # Retrieve the count label for this image (default to 0.0 if not found).
        count = self.labels.get(img_id, 0.0)
        # Return the image tensor and the count as a float tensor.
        return image, torch.tensor([count], dtype=torch.float)


# =============================================================================
# Helper Function for CSRNet: make_layers
# =============================================================================
def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    """
    Creates a sequential module based on a configuration list.

    Parameters:
        cfg (list): List containing numbers (number of output channels) and 'M' for MaxPool.
        in_channels (int): Number of input channels.
        batch_norm (bool): Whether to add BatchNorm layers.
        dilation (bool): If True, uses dilation in convolutions.

    Returns:
        nn.Sequential: The constructed layers.
    """
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if dilation:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# =============================================================================
# Model Class 1.0: CrowdCounter, ResNet18
# =============================================================================
class CrowdCounterResNet18(nn.Module):
    def __init__(self):
        """
        Initializes the CrowdCounter model using a pretrained ResNet18 backbone.
        """
        super().__init__()
        # Load a pretrained ResNet18 model with ImageNet weights.
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Build the feature extractor using ResNet18's initial layers.
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4
        )

        # Define the density estimation head as a series of convolutional layers.
        self.density_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

        # Initialize the weights of the density head layers.
        for m in self.density_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass through the model.

        Parameters:
            x (torch.Tensor): Input image tensor of shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: Predicted counts with shape [batch_size, 1].
        """
        # Extract features from the input using the pretrained layers.
        x = self.features(x)
        # Pass the features through the density head to obtain a density map.
        x = self.density_head(x)
        # Aggregate the density map over spatial dimensions (height and width) to get a scalar count.
        x = torch.sum(x, dim=[2, 3])
        return x  # Output shape: [batch_size, 1]

# =============================================================================
# Model Variant 1.1: CrowdCounter, ResNet50, Deeper model
# =============================================================================
class CrowdCounterResNet50(nn.Module):
    def __init__(self):
        """
        Initializes the CrowdCounter model with ResNet50 as the backbone.
        """
        super().__init__()
        # Load a pretrained ResNet50 from torchvision.
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Build the feature extractor from ResNet50 layers.
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4
        )
        # ResNet50's layer4 outputs 2048 channels.
        # Adjust the density head to account for this.
        self.density_head = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        # Initialize the weights of the density head layers.
        for m in self.density_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass: extracts features, produces a density map, then aggregates to a scalar count.
        """
        x = self.features(x)
        x = self.density_head(x)
        # Sum over the spatial dimensions to yield a count per image.
        x = torch.sum(x, dim=[2, 3])
        return x  # Output shape: [batch_size, 1]

# =============================================================================
# Model Variant 2: CSRNet
# =============================================================================
class CSRNet(nn.Module):
    def __init__(self):
        """
        Initializes the CSRNet model for crowd counting.
        CSRNet uses a VGG-16–like frontend and a dilated backend.
        """
        super(CSRNet, self).__init__()
        # Frontend configuration based on VGG-16.
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        # Backend configuration with dilated convolutions.
        self.backend_feat = [512, 512, 512, 256, 128, 64, 64]
        # Build the frontend and backend.
        self.frontend = make_layers(self.frontend_feat, in_channels=3, batch_norm=False, dilation=False)
        self.backend = make_layers(self.backend_feat, in_channels=512, batch_norm=False, dilation=True)
        # Output layer to produce a single-channel density map.
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)

    def forward(self, x):
        """
        Forward pass: process input through frontend, backend, then output layer.
        Aggregates the density map to return a scalar count per image.
        """
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        x = torch.sum(x, dim=[2, 3])
        return x

# =============================================================================
# Training Function (uses separate train and validation folders)
# =============================================================================
def train_our_model(root_directory=r"C:\Users\jm190\Desktop\jhu_crowd_v2.0", model_type="resnet50"):
    """
    Trains a crowd counting model using either the ResNet50 variant or CSRNet.

    Parameters:
        root_directory (str): Root directory containing 'train', 'val', and 'test' folders.
        model_type (str): Which model to use ("resnet50" or "csrnet").

    Returns:
        model (torch.nn.Module): The trained model.
    """
    # Define transforms for training and validation.
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load training and validation datasets from their respective folders.
    train_dataset = CountDataset(root_directory, split='train', transform=train_transform)
    val_dataset = CountDataset(root_directory, split='val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose the model variant based on the model_type parameter.
    if model_type.lower() == "resnet50":
        model = CrowdCounterResNet50().to(device)
    elif model_type.lower() == "csrnet":
        model = CSRNet().to(device)
    else:
        raise ValueError("Unsupported model type. Choose either 'resnet50' or 'csrnet'.")

    criterion = nn.L1Loss()  # Use Mean Absolute Error loss.
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    scaler = GradScaler()

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10  # Early stopping patience.

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, counts in train_loader:
            images = images.to(device, non_blocking=True)
            counts = counts.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, counts)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, counts in val_loader:
                images = images.to(device, non_blocking=True)
                counts = counts.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, counts)
                val_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    model.load_state_dict(torch.load("best_model.pth"))
    return model


def predict_people_in_image(model_path, image_path, device=None):
    """
    Loads the saved CrowdCounter model and predicts the number of people in the given image.

    Parameters:
        model_path (str): Path to the saved model file (e.g., "best_model.pth").
        image_path (str): Path to the image file on which to perform inference.
        device (torch.device, optional): Device to run the model on. If None, it will automatically
                                         use CUDA if available, otherwise CPU.

    Returns:
        float: The predicted count of people in the image.
    """
    # If no device is provided, select CUDA if available, otherwise use CPU.
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transform pipeline for inference.
    # Note: We use only deterministic transforms (no random augmentation).
    from torchvision import transforms
    from PIL import Image
    inference_transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize image to the size expected by the model.
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats.
    ])

    # Initialize the model and load the saved weights.
    model = CrowdCounterResNet50().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode.

    # Load and preprocess the image.
    image = Image.open(image_path).convert("RGB")
    image_tensor = inference_transform(image)
    # Add a batch dimension: shape becomes [1, channels, height, width]
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Perform inference without computing gradients.
    with torch.no_grad():
        # Optionally, you can use autocast for mixed precision if desired:
        # from torch.amp import autocast
        # with autocast('cuda'):
        output = model(image_tensor)

    # The model returns a tensor of shape [1, 1]. Extract the scalar value.
    predicted_count = output.item()
    return predicted_count

# =============================================================================
# Main Execution Block
# =============================================================================
if __name__ == "__main__":
    # Path to the dataset configuration file (may need editing if using a new dataset).
    data_yaml = "overhead_data.yaml"  # Not directly used in this snippet.

    # Root directory where the dataset is located.
    root_directory = r"C:\Users\jm190\Desktop\jhu_crowd_v2.0"  # Update if your dataset location changes.

    # Choose the model type: either "resnet50" for a deeper ResNet backbone or "csrnet" for the CSRNet model.
    # trained_model = train_our_model(root_directory, model_type="resnet50")

    model_path = "best_model.pth"
    image_path = r"C:\Users\jm190\Desktop\jhu_crowd_v2.0\test\images\2156.jpg"  # Update this path accordingly.

    # Get the prediction.
    count = predict_people_in_image(model_path, image_path)
    print(f"Predicted number of people: {count:.2f}")