import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        A simple feedforward neural network.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of neurons in the hidden layer.
            output_dim (int): Number of output classes.
        """
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(),                        
            nn.Linear(hidden_dim, output_dim) 
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output predictions.
        """
        return self.net(x)

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    input_dim = 4      
    hidden_dim = 8    
    output_dim = 3     
    learning_rate = 0.01

    # Initialize the network, loss function, and optimizer
    model = SimpleNN(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()  # For classification tasks
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Example data (batch size = 5, input features = 4)
    inputs = torch.rand(5, input_dim)        # Random input data
    targets = torch.tensor([0, 1, 2, 1, 0])  # Corresponding labels (0, 1, or 2)

    # Training step
    model.train()  
    outputs = model(inputs)                  # Forward pass
    loss = criterion(outputs, targets)      # Compute loss
    optimizer.zero_grad()                    # Zero out gradients
    loss.backward()                          # Backpropagation
    optimizer.step()                         # Update parameters

    # Print results
    print("Inputs:\n", inputs)
    print("Targets:\n", targets)
    print("Predicted Outputs:\n", outputs)
    print("Loss:\n", loss.item())