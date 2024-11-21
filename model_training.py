# model_training.py

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GATConv

def train_gat_model(pyg_graph, df_economy):
    # Define the GAT model architecture
    class GATModel(torch.nn.Module):
        def __init__(self, num_node_features, num_classes, num_heads=4):
            super(GATModel, self).__init__()
            self.conv1 = GATConv(num_node_features, 128, heads=num_heads)  # First GAT layer
            self.conv2 = GATConv(128 * num_heads, 64, heads=num_heads)  # Second GAT layer
            self.fc = torch.nn.Linear(64 * num_heads, num_classes)  # Fully connected layer for output

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.elu(x)  # Exponential Linear Unit
            x = F.dropout(x, training=self.training)

            x = self.conv2(x, edge_index)
            x = F.elu(x)  # Exponential Linear Unit
            x = F.dropout(x, training=self.training)

            x = self.fc(x)
            return x

    # Create a mapping from ideology code to class index
    unique_labels = sorted(df_economy['cmp_code'].unique())
    ideology_to_class = {label: idx for idx, label in enumerate(unique_labels)}

    # Prepare the target labels for the pyg_graph
    pyg_graph.y = torch.tensor(
        [ideology_to_class.get(label.item(), ideology_to_class[unique_labels[-1]]) for label in pyg_graph.y],
        dtype=torch.long
    )

    # Ensure that the maximum index in pyg_graph.y is less than the number of classes
    num_classes = len(ideology_to_class)
    assert (pyg_graph.y.max().item() < num_classes), f"Target {pyg_graph.y.max().item()} is out of bounds for number of classes {num_classes}."

    # Define optimizer and loss function
    model = GATModel(num_node_features=pyg_graph.x.size(1), num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Move the model and data to the correct device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    pyg_graph = pyg_graph.to(device)

    # Split the dataset into train and test (you can define your own split here)
    train_mask = torch.zeros(pyg_graph.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(pyg_graph.num_nodes, dtype=torch.bool)

    # Example split (80% train, 20% test)
    num_train = int(0.8 * pyg_graph.num_nodes)
    train_mask[:num_train] = 1
    test_mask[num_train:] = 1

    # Attach masks to the graph
    pyg_graph.train_mask = train_mask
    pyg_graph.test_mask = test_mask

    # Training loop
    model.train()
    for epoch in range(200):  # Train for 200 epochs
        optimizer.zero_grad()

        # Forward pass
        out = model(pyg_graph.x, pyg_graph.edge_index)

        # Compute loss (only on nodes with ideology labels)
        loss = criterion(out[pyg_graph.train_mask], pyg_graph.y[pyg_graph.train_mask])

        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        # Print training loss
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model
