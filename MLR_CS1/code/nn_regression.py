import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from helpers.loss import CustomLoss
from helpers.metrics import compute_mse, compute_position_error, compute_rotation_error
from datasets import prepare_dataset
from visualizer import plot_linear_trajectory


def convert_to_tensor(data, device="cpu"):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).float().to(device)
    return torch.tensor(data, dtype=torch.float32).to(device)

class MLP(nn.Module):
    def __init__(self, input_size=6, hidden_sizes=[128, 64], output_size=6):
        """
        Initialize a Multi-Layer Perceptron (MLP) neural network.

        Args:
            input_size (int, optional): The number of input features. 
            hidden_sizes (list, optional): A list of integers representing the number of neurons 
                in each hidden layer. 
            output_size (int, optional): The number of output neurons. 

        Description:
            Constructs an MLP with the specified architecture. The network consists of:
            - An input layer that maps from input_size to the first hidden layer
            - Hidden layers with ReLU activation functions between them
            - An output layer with no activation function
            
            All layers are stored in a Sequential container accessible via self.network.
        """
        super(MLP, self).__init__()
        layers = []

        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers) 

    def forward(self, x):
        return self.network(x)

    def fit(
         self,
         X_train,
         y_train,
         lr=0.001,
         batch_size=32,
         epochs=100,
         device="gpu",
     ):
   
         # Convert to tensors
         X_train_tensor = convert_to_tensor(X_train, device)
         y_train_tensor = convert_to_tensor(y_train, device)
         
         # Create data loaders
         train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
     
         # Initialize loss and optimizer
         criterion = CustomLoss(position_weight=1.0, rotation_weight=1.0)
         optimizer = torch.optim.Adam(self.parameters(), lr=lr)

         # Training loop
         self.train()
         for epoch in range(epochs):
             for X_batch, y_batch in train_loader:
                 optimizer.zero_grad()
                 y_pred = self(X_batch)
                 loss = criterion(y_pred, y_batch)
                 loss.backward()
                 optimizer.step()

         return self

    def predict(self, X, device="cuda"):
        X_tensor = convert_to_tensor(X, device)
        self.eval()
        with torch.no_grad():
            y_pred = self(X_tensor).cpu().numpy()
        return y_pred   

if __name__ == "__main__":

    # Load and prepare data
    X_train, X_test, y_train, y_test = prepare_dataset("data/ur10_dataset.csv")

    model = MLP().to("cuda")

    # Train model
    model.fit(
        X_train.values,
        y_train.values,
        lr=0.001,
        batch_size=32,
        epochs=100,
        device="cuda",
    )

    plot_linear_trajectory(model)