- **Custom Dataset Class:**  
  `CountDataset` loads images and their corresponding labels from a folder structure and performs error checking.

- **Model Architectures:**  
  - **CrowdCounterResNet18:** Uses a pretrained ResNet18 backbone.  
  - **CrowdCounterResNet50:** A deeper variant using a pretrained ResNet50 backbone.  
  - **CSRNet:** A specialized model with a VGG-16–like frontend and dilated convolution backend.

- **Training Pipeline:**  
  The training script applies data augmentation, uses mixed precision (with `GradScaler` and `autocast`), employs a learning rate scheduler (`ReduceLROnPlateau`), and includes early stopping.

- **Inference Function:**  
  The `predict_people_in_image` function loads a saved model checkpoint and predicts the crowd count for a given image.

## Requirements

- Python 3.6+
- [PyTorch](https://pytorch.org/) (v1.7 or higher recommended)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- [Pillow](https://python-pillow.org/) (for image processing)

Install the dependencies using pip:

```bash
pip install torch torchvision pillow
