import torch
import cv2
from PIL import Image
# from torchvision.transforms import ToTensor
from net.fai_gcn import Model # Import the FAIGCN model class from the repository

# Load the pre-trained model weights
model_path = 'work_dir/recognition/kinetics_skeleton/FAI_GCN/epoch500_model.pt'  # Replace with the path to the downloaded model weights
state_dict = torch.load(model_path, map_location=torch.device('cpu'))  # Load the model weights

# Instantiate the model class
model = Model(32, 64, **kwargs=)  # Replace with the appropriate model class and number of classes

# Load the model weights into the model
model.load_state_dict(state_dict)
model.eval()

# Load and preprocess the input image
image_path = 'path/to/test_image.jpg'  # Replace with the path to your test image
image = Image.open(image_path).convert('RGB')
image = image.resize((256, 256))  # Resize the image to the input size of the model
# image = ToTensor()(image)  # Convert the image to a PyTorch tensor

# Perform inference with the model
with torch.no_grad():
    # Add a batch dimension to the input image
    image = image.unsqueeze(0)
    
    # Forward pass through the model
    output = model(image)
    
    # Get the predicted class and confidence scores
    predicted_class = torch.argmax(output, dim=1).item()
    confidence_scores = torch.softmax(output, dim=1).squeeze().tolist()

# Print the predicted class and confidence scores
print("Predicted class:", predicted_class)
print("Confidence scores:", confidence_scores)
