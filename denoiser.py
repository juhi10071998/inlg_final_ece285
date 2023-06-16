import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define the DirectDecoder architecture
class DirectDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(DirectDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# Define the training loop
def train_model(model, train_loader, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        if (epoch+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))

# Example usage
# Assuming you have demonized_embeddings and text_embeddings as input data

# Convert the data to PyTorch tensors
demonized_embeddings = torch.tensor(demonized_embeddings, dtype=torch.float32)
text_embeddings = torch.tensor(text_embeddings, dtype=torch.float32)

# Create a TensorDataset
dataset = TensorDataset(demonized_embeddings, text_embeddings)

# Define the batch size for the data loader
batch_size = 4

# Create the data loader
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the dimensions of the demonized embedding and text embedding
input_dim = 512
output_dim = 768

# Define the dimensions of the hidden layer in the decoder
hidden_dim = 256

# Instantiate the DirectDecoder model
model = DirectDecoder(input_dim, output_dim, hidden_dim)

# Train the model
num_epochs = 50
learning_rate = 0.001
train_model(model, train_loader, num_epochs, learning_rate)

# Example inference
sample_input = torch.tensor([[0.1, 0.2, 0.3, ...]])  # Replace with your own input of size 512
predicted_output = model(sample_input)
print("Predicted text embedding:", predicted_output)
