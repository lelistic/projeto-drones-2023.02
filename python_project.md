# Project Description: Optical Flow Estimation and Model Management

This project aims to bridge the gap between real-world scenarios and simulations by estimating optical flow from consecutive frames of images captured by drones. The proposed solution includes computing optical flow vectors, designing a custom neural network architecture for optical flow estimation, and managing the trained model for future use and updates.

## Project Structure:

**Requirements:**

Store the required library dependencies in a file named requirements.txt.

**Script to Compute Optical Flow (compute_optical_flow.py):**

Compute optical flow vectors between two consecutive frames using OpenCV's Lucas-Kanade method.

**Script for Custom Neural Network Architecture (custom_network.py):**

Define a custom neural network architecture optimized for optical flow estimation using PyTorch.

**Script to Train the Neural Network (train_network.py):**

Train the custom neural network using training data, loss functions, and optimization techniques.

**Script to Save a Trained Model (save_model.py):**

Save the trained neural network model's parameters for future use.

**Script to Load and Update a Model (load_and_update_model.py):**

Load a saved model and update it with new data to adapt to changing scenarios.

**Orchestration Script (orchestrate.py):**

An orchestration script to manage the entire process from computing optical flow to saving, loading, and updating the model.

1. **requirements.txt:**
   Save this file as `requirements.txt` in your project directory.
   ```
   torch==1.9.0
   torchvision==0.10.0
   opencv-python==4.5.3.56
   ```

2. **compute_optical_flow.py:**
   Save this script as `compute_optical_flow.py` in your project directory.
   ```python
   import cv2
   import numpy as np

   def compute_optical_flow(frame1, frame2):
       # Parameters for Lucas-Kanade optical flow
       lk_params = dict(
           winSize=(15, 15),
           maxLevel=2,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
       )

       # Calculate optical flow using Lucas-Kanade
       optical_flow = cv2.calcOpticalFlowPyrLK(frame1, frame2, None, None, **lk_params)
       flow_vectors = optical_flow[0]

       return flow_vectors

   # Load consecutive frames
   frame1 = cv2.imread('frame1.jpg', cv2.IMREAD_GRAYSCALE)
   frame2 = cv2.imread('frame2.jpg', cv2.IMREAD_GRAYSCALE)

   # Compute optical flow vectors
   flow_vectors = compute_optical_flow(frame1, frame2)
   print(flow_vectors)
   ```

3. **custom_network.py:**
   Save this script as `custom_network.py` in your project directory.
   ```python
   import torch
   import torch.nn as nn

   class OpticalFlowEstimationNetwork(nn.Module):
       def __init__(self):
           super(OpticalFlowEstimationNetwork, self).__init__()

           # Define layers for optical flow estimation
           self.fc1 = nn.Linear(2, 128)  # Input: 2 components of flow vectors
           self.fc2 = nn.Linear(3, 128)  # Input: magnitude, angle, and one more feature
           self.fc3 = nn.Linear(256, 2)  # Output: 2 components of predicted flow vectors

       def forward(self, flow_vectors, magnitude, angle):
           flow_vectors = self.fc1(flow_vectors)
           other_features = torch.cat((magnitude.unsqueeze(-1), angle.unsqueeze(-1)), dim=1)
           other_features = self.fc2(other_features)

           combined_features = torch.cat((flow_vectors, other_features), dim=1)
           predicted_flow = self.fc3(combined_features)

           return predicted_flow
   ```

4. **train_network.py:**
   Save this script as `train_network.py` in your project directory.
   ```python
   import torch
   import torch.optim as optim
   import custom_network

   # Load training data
   flow_vectors = torch.randn(100, 2)  # Sample flow vectors
   magnitude = torch.randn(100)  # Sample magnitudes
   angle = torch.randn(100)  # Sample angles
   target_flow = torch.randn(100, 2)  # Ground truth flow vectors

   # Instantiate the custom neural network
   network = custom_network.OpticalFlowEstimationNetwork()

   # Loss function and optimizer
   criterion = torch.nn.MSELoss()
   optimizer = optim.SGD(network.parameters(), lr=0.01)

   # Training loop
   for epoch in range(100):
       optimizer.zero_grad()
       predicted_flow = network(flow_vectors, magnitude, angle)
       loss = criterion(predicted_flow, target_flow)
       loss.backward()
       optimizer.step()
       print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
   ```

To run this project, make sure you have the required libraries installed using the `requirements.txt` file:

```
pip install -r requirements.txt
```

After that, you can run the individual scripts:

```
python compute_optical_flow.py
```

```
python train_network.py
```

--------------------------------------

Scripts to save a trained neural network model and load it for future use, as well as a script to update the loaded model with new data:

1. **Script to Save a Trained Model (`save_model.py`):**

Save this script as `save_model.py` in your project directory.

```python
import torch
import custom_network

# Instantiate and load the trained model
network = custom_network.OpticalFlowEstimationNetwork()
checkpoint = {'model_state_dict': network.state_dict()}
torch.save(checkpoint, 'trained_model.pth')
print("Trained model saved.")
```

Run this script after training your model to save its parameters to a file named `trained_model.pth`.

2. **Script to Load and Update a Model (`load_and_update_model.py`):**

Save this script as `load_and_update_model.py` in your project directory.

```python
import torch
import custom_network

# Load the model architecture
loaded_network = custom_network.OpticalFlowEstimationNetwork()

# Load the saved model parameters
checkpoint = torch.load('trained_model.pth')
loaded_network.load_state_dict(checkpoint['model_state_dict'])
print("Trained model loaded.")

# Prepare new data for updating the model
new_flow_vectors = torch.randn(50, 2)  # New sample flow vectors
new_magnitude = torch.randn(50)  # New sample magnitudes
new_angle = torch.randn(50)  # New sample angles
new_target_flow = torch.randn(50, 2)  # New ground truth flow vectors

# Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(loaded_network.parameters(), lr=0.01)

# Update the loaded model with new data
for epoch in range(50):
    optimizer.zero_grad()
    new_predicted_flow = loaded_network(new_flow_vectors, new_magnitude, new_angle)
    new_loss = criterion(new_predicted_flow, new_target_flow)
    new_loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Updated Loss: {new_loss.item():.4f}')
```

Run the `load_and_update_model.py` script to load the previously trained model, update it with new data, and save the updated model parameters back to the `trained_model.pth` file.



---------------------------------

**Orchestration Script (orchestrate.py):**

Save this script as orchestrate.py in your project directory.

```python
import compute_optical_flow
import custom_network
import torch
import torch.optim as optim

# Load consecutive frames
frame1 = cv2.imread('frame1.jpg', cv2.IMREAD_GRAYSCALE)
frame2 = cv2.imread('frame2.jpg', cv2.IMREAD_GRAYSCALE)

# Compute optical flow vectors
flow_vectors = compute_optical_flow.compute_optical_flow(frame1, frame2)

# Instantiate and load the trained model
network = custom_network.OpticalFlowEstimationNetwork()
checkpoint = torch.load('trained_model.pth')
network.load_state_dict(checkpoint['model_state_dict'])
print("Trained model loaded.")

# Prepare new data for updating the model
new_flow_vectors = torch.randn(50, 2)  # New sample flow vectors
new_magnitude = torch.randn(50)  # New sample magnitudes
new_angle = torch.randn(50)  # New sample angles
new_target_flow = torch.randn(50, 2)  # New ground truth flow vectors

# Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.SGD(network.parameters(), lr=0.01)

# Update the loaded model with new data
for epoch in range(50):
    optimizer.zero_grad()
    new_predicted_flow = network(new_flow_vectors, new_magnitude, new_angle)
    new_loss = criterion(new_predicted_flow, new_target_flow)
    new_loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Updated Loss: {new_loss.item():.4f}')

# Save the updated model
checkpoint = {'model_state_dict': network.state_dict()}
torch.save(checkpoint, 'updated_model.pth')
print("Updated model saved.")
```

This orchestrate.py script encapsulates the entire process, from computing optical flow to saving, loading, and updating the model. Customize the file paths, data dimensions, and training parameters based on your dataset and requirements. Then, run this script to execute the complete workflow.
