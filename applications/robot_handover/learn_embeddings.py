import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Define the function to plot sequences
def plot_sequences(original, reconstructed, num_samples=10):
    """
    Plots the original and reconstructed sequences.

    Parameters:
    - original (torch.Tensor): The original sequences (shape: [batch_size, seq_len, input_dim]).
    - reconstructed (torch.Tensor): The reconstructed sequences (shape: [batch_size, seq_len, input_dim]).
    - num_samples (int): Number of samples to plot.
    """
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().detach().numpy()
    
    num_samples = min(num_samples, original.shape[0])
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, 3 * num_samples))
    
    for i in range(num_samples):
        axes[i, 0].plot(original[i])
        axes[i, 0].set_title(f'Original Sequence {i}')
        axes[i, 0].set_xlabel('Time Step')
        axes[i, 0].set_ylabel('Value')
        
        axes[i, 1].plot(reconstructed[i])
        axes[i, 1].set_title(f'Reconstructed Sequence {i}')
        axes[i, 1].set_xlabel('Time Step')
        axes[i, 1].set_ylabel('Value')
    
    plt.tight_layout()
    plt.show()

def conv1d_output_size(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1




import torch.nn as nn

class Conv1dAutoencoder(nn.Module):
    def __init__(self, latent_size=32):
        super(Conv1dAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=16, kernel_size=12, stride=2, padding=5),
            nn.ReLU(True),
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=12, stride=2, padding=5),
            nn.ReLU(True),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=12, stride=2, padding=5),
            nn.LeakyReLU(True)
        )
        
        # Determine output size after convolutional layers
        self.conv_out_size = self._get_conv_output_size(input_size=50)
        
        self.fc1 = nn.Linear(self.conv_out_size, 512)
        self.fc2 = nn.Linear(512, latent_size)

        self.map = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.LeakyReLU(True),
            nn.Linear(512, self.conv_out_size),
            nn.LeakyReLU(True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=64, kernel_size=12, stride=2, padding=5, output_padding=1),
            nn.LeakyReLU(True),
            nn.ConvTranspose1d(in_channels=64, out_channels=16, kernel_size=12, stride=2, padding=5, output_padding=1),
            nn.LeakyReLU(True),
            nn.ConvTranspose1d(in_channels=16, out_channels=6, kernel_size=12, stride=2, padding=5, output_padding=1),
        )

    def _get_conv_output_size(self, input_size):
        size = input_size
        size = conv1d_output_size(size, kernel_size=12, stride=2, padding=5)
        size = conv1d_output_size(size, kernel_size=12, stride=2, padding=5)
        size = conv1d_output_size(size, kernel_size=12, stride=2, padding=5)
        return size * 64  # 64 is the number of channels after the last convolution
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.map(x)
        x = x.view(x.size(0), 64, -1)  # Reshape
        x = self.decoder(x)
        return x[:, :, :50]  # Crop to match the input size
    

def unit_ball_loss(embedding):
    """
    Compute the unit ball loss for the embedding tensor.
    This loss penalizes deviations from unit norm.
    """
    return torch.sum((torch.norm(embedding, p=2, dim=1) - 1) ** 2)



# Load the dataset
data = np.load('handovers.npy')
# Normalize the dataset
mean = data.mean(axis=(0, 1), keepdims=True)
std = data.std(axis=(0, 1), keepdims=True)
normalized_data = (data - mean) / std

# Convert to PyTorch tensor
data_tensor = torch.tensor(normalized_data, dtype=torch.float32)

# Create a DataLoader
batch_size = 64
dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






LATENT_SIZE = 8



# Instantiate the model and move it to the GPU
model = Conv1dAutoencoder(latent_size=LATENT_SIZE).to(device)
print(model)


import torch.optim as optim

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for data in dataloader:
        inputs = data[0].permute(0, 2, 1).to(device)  # Change shape and move to GPU
        
       # Forward pass
        latent_embedding = model.encoder(inputs).view(inputs.size(0), -1)
        outputs = model(inputs)

        reconstruction_loss = criterion(outputs, inputs)
        ub_loss = unit_ball_loss(latent_embedding)

        loss = reconstruction_loss + 1000 * ub_loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    print(f'Reconstruction Loss: {reconstruction_loss.item():.4f}, Unit Ball Loss: {ub_loss.item():.4f}')


# Switch to evaluation mode
model.eval() 

# Get a batch of data
inputs = next(iter(dataloader))[0].permute(0, 2, 1).to(device)

# Forward pass
with torch.no_grad():
    outputs = model(inputs)

# Move data back to CPU for printing
inputs = inputs.cpu()
outputs = outputs.cpu()

# Plot original vs reconstructed sequences
plot_sequences(inputs.permute(0, 2, 1), outputs.permute(0, 2, 1))

# Assume 'model' is your instance of Conv1dAutoencoder
torch.save(model.state_dict(), 'autoencoder_model.pth')










# Print the results
# print("Original:", inputs[0].permute(1, 0))  # Change shape back to (50, 4) for printing
# print("Reconstructed:", outputs[0].permute(1, 0))  # Change shape back to (50, 4) for printing



import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load and normalize the dataset
data = np.load('handovers.npy')
mean = data.mean(axis=(0, 1), keepdims=True)
std = data.std(axis=(0, 1), keepdims=True)

# Correct the shapes of mean and std for broadcasting
mean = np.transpose(mean, (1, 0, 2))  # Shape: (4, 1, 50)
std = np.transpose(std, (1, 0, 2))    # Shape: (4, 1, 50)

# Normalize the dataset
normalized_data = (data - mean) / std

# Transpose data to match Conv1d expected input shape
normalized_data = np.transpose(normalized_data, (0, 2, 1))

# Create DataLoader
batch_size = 32
data_loader = DataLoader(TensorDataset(torch.tensor(normalized_data, dtype=torch.float32)), batch_size=batch_size, shuffle=False)

# Initialize the autoencoder model and load it onto the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Conv1dAutoencoder(latent_size=LATENT_SIZE).to(device)
model.load_state_dict(torch.load('autoencoder_model.pth'))  # Load the trained model

model.eval()  # Set the model to evaluation mode

# Initialize array for storing embeddings and reconstructions
embeddings = np.empty((data.shape[0], LATENT_SIZE))  # Assuming latent_size=32
reconstructed_data = np.empty_like(normalized_data)

# Process data in batches
with torch.no_grad():
    for batch_idx, (batch_data,) in enumerate(data_loader):
        batch_data = batch_data.to(device)
        # Get embeddings from the encoder
        batch_embeddings = model.encode(batch_data)
        embeddings[batch_idx * batch_size:(batch_idx + 1) * batch_size] = batch_embeddings.cpu().numpy()

        # Reconstruct data from the decoder
        outputs = model(batch_data)
        reconstructed_data[batch_idx * batch_size:(batch_idx + 1) * batch_size] = outputs.cpu().numpy()

# Unnormalize the reconstructions
reconstructed_data = np.transpose(reconstructed_data, (0, 2, 1))  # Shape: (1500, 50, 4)
unnormalized_reconstructions = reconstructed_data * std + mean

# Save the unnormalized data and embeddings
np.save('handover_reconstructions.npy', unnormalized_reconstructions)
np.save('handover_embeddings.npy', embeddings)