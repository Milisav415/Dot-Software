import os
import csv
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
from ultralytics import YOLO
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

# Define a custom Dataset that reads images and count labels
class CountDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        root_dir: the folder containing 'train', 'val', 'test'
        split: one of 'train', 'val', 'test'
        transform: image transformations (e.g. resizing, normalization)
        """
        self.split_dir = os.path.join(root_dir, split)
        self.images_dir = os.path.join(self.split_dir, "images")
        self.labels_file = os.path.join(self.split_dir, "image_labels.txt")
        self.transform = transform

        # Read the image-level labels file
        # Expected format per line: image_id, count, <other fields>
        self.labels = {}  # mapping from image id (string) to count (integer)
        with open(self.labels_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                img_id, count = row[0].strip(), row[1].strip()
                self.labels[img_id] = float(count)

        # Get a list of image files that correspond to the labels (assumes images have a .jpg extension)
        self.image_files = []
        for img_id in self.labels.keys():
            # You might need to adjust the extension if your images have a different format
            img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
            if os.path.exists(img_path):
                self.image_files.append(img_path)
            else:
                print(f"Warning: Image file {img_path} not found.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        # Extract image id (filename without extension)
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        # Open image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Get the count for the image
        count = self.labels.get(img_id, 0.0)
        # Return image tensor and count as a float tensor
        return image, torch.tensor([count], dtype=torch.float)

# Define a simple regression model based on a pretrained ResNet18.
class CountRegressionModel(nn.Module):
    def __init__(self):
        super(CountRegressionModel, self).__init__()
        # Load a pretrained ResNet18 model
        self.backbone = models.resnet18(pretrained=True)
        # Replace the final fully connected layer (originally 512 -> 1000)
        self.backbone.fc = nn.Linear(512, 1)  # output a single regression value

    def forward(self, x):
        return self.backbone(x)


def predict_count(model, image_or_path, device, transform=None):
    """
    Predict a count from a single image using the provided model.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        image_or_path (str or PIL.Image.Image): Either a path to the image file or a PIL Image.
        device (torch.device): Device on which to perform inference (e.g., torch.device('cuda') or torch.device('cpu')).
        transform (callable, optional): A function/transform that converts the image to a tensor
                                        and applies any necessary preprocessing. Default is None.

    Returns:
        float: The predicted count (a scalar value).
    """
    # Load image if a path is provided
    if isinstance(image_or_path, str):
        image = Image.open(image_or_path).convert("RGB")
    else:
        image = image_or_path

    # Apply transformation if provided
    if transform:
        image = transform(image)

    # Add a batch dimension
    image = image.unsqueeze(0).to(device)

    # Set model to evaluation mode and perform inference
    model.eval()
    with torch.no_grad():
        output = model(image)

    # Assuming the model outputs a tensor of shape [1, 1], convert it to a Python float
    return output.item()


def count_people_yolov8(image_or_path, conf_threshold=0.1, device="cpu"):
    """
    Counts people in an image using a pretrained YOLOv8 model.

    Args:
        image_or_path (str or PIL.Image.Image): Image path or PIL image.
        conf_threshold (float): Confidence threshold to filter detections.
        device (str): Device to run inference on ("cpu" or "cuda").

    Returns:
        int: Number of detected people.
        np.ndarray: The image with drawn bounding boxes.
    """
    # Load YOLOv8 model pretrained on COCO (which has the person class)
    model = YOLO("yolov8n.pt")  # YOLOv8 nano variant pretrained on COCO
    model.to(device)

    # Load image
    if isinstance(image_or_path, str):
        # Use OpenCV to read the image
        image = cv2.imread(image_or_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_or_path}")
    else:
        # Convert PIL image to OpenCV format
        image = cv2.cvtColor(np.array(image_or_path), cv2.COLOR_RGB2BGR)

    # Perform inference
    results = model(image, device=device)
    detections = results[0].boxes  # Boxes for first (and only) image

    # Filter detections by confidence and class (person usually has class index 0 in COCO)
    person_count = 0
    for box, conf, cls in zip(detections.xyxy, detections.conf, detections.cls):
        if conf >= conf_threshold and int(cls) == 0:  # COCO class 0 is person
            person_count += 1
            # Draw bounding box on image (green)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"person {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return person_count, image

def train_our_owen():
    # Set up image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # resize images as needed (match your model's expected input)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # the fucking path
    root_directory = "C:\\Users\\jm190\\Desktop\\jhu_crowd_v2.0"  # adjust path as needed

    # model = CountRegressionModel()
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_dataset = CountDataset(root_directory, split='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

    # Training loop (basic example)
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, counts in train_loader:
            images = images.to(device)
            counts = counts.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, counts)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # (Optionally) Save the trained model
    # torch.save(model.state_dict(), "count_regression_model.pth")
    image_path = 'C:\\Users\\jm190\\Desktop\\jhu_crowd_v2.0\\test\\images\\1222.jpg'
    print(f'I say: {predict_count(model, image_path, device, transform=transform)} but its actually 409')

# I know this looks ugly but shutup I`m not getting paid
if __name__ == "__main__":
    # train_our_owen()
    image_path = 'C:\\Users\\jm190\\Desktop\\jhu_crowd_v2.0\\test\\images\\1222.jpg'

    count, image = count_people_yolov8(image_or_path=image_path)

    print(f'The model thinks there are: {count}\n')

