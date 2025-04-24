import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Check if CUDA is available
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Load the pre-trained DeepLabV3 model (trained on COCO)
model = deeplabv3_resnet50(pretrained=True, progress=True)
model = model.to(device)
model.eval()

# Define the image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((520, 520)),  # Resize to a size compatible with DeepLabV3
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
image_path = "/mnt/e_disk/ch24s016/ee6180_assignment1/pix2pix/datasets/2/ADEChallengeData2016/images/training/ADE_train_00019788.jpg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found at {image_path}")

try:
    input_image = Image.open(image_path).convert("RGB")
except Exception as e:
    raise ValueError(f"Error loading image: {e}")

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0).to(device)  # Add batch dimension

# Run the model
with torch.no_grad():
    output = model(input_batch)['out'][0]
output_predictions = output.argmax(0)  # Get the predicted class for each pixel
output_predictions = output_predictions.cpu().numpy()

# Define a color map for COCO classes (21 classes, including background)
# Based on PASCAL VOC/COCO labels used by DeepLabV3
color_map = {
    0: [0, 0, 0],        # Background (black)
    1: [128, 0, 0],      # Person (red)
    2: [0, 128, 0],      # Bicycle (green)
    3: [128, 128, 0],    # Car (yellow)
    4: [0, 0, 128],      # Motorcycle (blue)
    5: [128, 0, 128],    # Airplane (purple)
    6: [0, 128, 128],    # Bus (cyan)
    7: [128, 128, 128],  # Train (gray)
    8: [64, 0, 0],       # Truck (dark red)
    9: [192, 0, 0],      # Boat (bright red)
    10: [64, 128, 0],    # Traffic light (lime)
    11: [192, 128, 0],   # Fire hydrant (orange)
    12: [64, 0, 128],    # Stop sign (violet)
    13: [192, 0, 128],   # Parking meter (magenta)
    14: [64, 128, 128],  # Bench (teal)
    15: [192, 128, 128], # Bird (pink)
    16: [0, 64, 0],      # Cat (dark green)
    17: [128, 64, 0],    # Dog (brown)
    18: [0, 192, 0],     # Horse (bright green)
    19: [128, 192, 0],   # Sheep (light yellow)
    20: [0, 64, 128],    # Cow (dark blue)
}

# Map predictions to colors
height, width = output_predictions.shape
colored_segmentation = np.zeros((height, width, 3), dtype=np.uint8)
for class_id, color in color_map.items():
    colored_segmentation[output_predictions == class_id] = color

# Save and display the segmentation map
output_path = 'segmentation_map.png'
plt.imsave(output_path, colored_segmentation)
print(f"Segmentation map saved as {output_path}")

# Optional: Display the original and segmented images side by side
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(input_image)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title("Segmentation Map")
plt.imshow(colored_segmentation)
plt.axis('off')
plt.show()