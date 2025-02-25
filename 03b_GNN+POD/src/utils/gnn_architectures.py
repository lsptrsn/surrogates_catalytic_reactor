import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv, ResGatedGraphConv
import torch.nn as nn
import ipdb
import random
import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

seed=1
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
rs = RandomState(MT19937(SeedSequence(seed)))


class GCN(torch.nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, window_size=1):
        """
        Initialize the GCN model.

        Parameters:
        - node_feature_dim (int): Dimensionality of node features.
        - hidden_dim (int): Dimensionality of hidden layers.
        """
        super(GCN, self).__init__()
        self.window_size = window_size

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GCN layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim*self.window_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feature_dim)
        )

    def forward(self, data):
        """
        Forward pass of the GCN model.

        Parameters:
        - data (torch_geometric.data.Data): Input graph data.

        Returns:
        - x (torch.Tensor): Predicted node features.
        """
        x, edge_index = data.x, data.edge_index

        # Encoder
        x = self.encoder(x)

        # Processor
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Decoder
        x = torch.chunk(x, self.window_size, dim=0)
        x = torch.cat(x, dim=1)
        x = self.decoder(x)
        x = F.sigmoid(x)

        return x

class GAT(torch.nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, window_size=1):
        """
        Initialize the GAT model.

        Parameters:
        - node_feature_dim (int): Dimensionality of node features.
        - hidden_dim (int): Dimensionality of hidden layers.
        - window_size (int): Number of previous snapshots to consider for prediction.
        """
        super(GAT, self).__init__()
        self.window_size = window_size

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GAT layers
        self.conv1 = GATConv(hidden_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim*self.window_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feature_dim)
        )

    def forward(self, data):
        """
        Forward pass of the GAT model.

        Parameters:
        - data (torch_geometric.data.Data): Input graph data.

        Returns:
        - x (torch.Tensor): Predicted node features.
        """
        x, edge_index = data.x, data.edge_index

        # Encoder
        x = self.encoder(x)

        # Processor
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Decoder
        x = torch.chunk(x, self.window_size, dim=0)
        x = torch.cat(x, dim=1)
        x = self.decoder(x)
        x = F.sigmoid(x)

        return x

class GGNN(torch.nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, window_size=1):
        """
        Initialize the GGNN model.

        Parameters:
        - node_feature_dim (int): Dimensionality of node features.
        - hidden_dim (int): Dimensionality of hidden layers.
        """
        super(GGNN, self).__init__()
        self.window_size = window_size

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GGNN layer
        self.conv1 = GatedGraphConv(hidden_dim, 5)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim*self.window_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feature_dim)
        )

    def forward(self, data):
        """
        Forward pass of the GGNN model.

        Parameters:
        - data (torch_geometric.data.Data): Input graph data.

        Returns:
        - x (torch.Tensor): Predicted node features.
        """
        x, edge_index = data.x, data.edge_index

        # Encoder
        x = self.encoder(x)

        # Processor
        x = F.relu(self.conv1(x, edge_index))
        
        # Decoder
        x = torch.chunk(x, self.window_size, dim=0)
        x = torch.cat(x, dim=1)
        x = self.decoder(x)
        x = F.sigmoid(x)

        return x

class RGGNN(torch.nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, window_size=1):
        """
        Initialize the RGGNN model.

        Parameters:
        - node_feature_dim (int): Dimensionality of node features.
        - hidden_dim (int): Dimensionality of hidden layers.
        """
        super(RGGNN, self).__init__()
        self.window_size = window_size

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # RGGNN layers
        self.conv1 = ResGatedGraphConv(hidden_dim, hidden_dim)
        self.conv2 = ResGatedGraphConv(hidden_dim, hidden_dim)
        self.conv3 = ResGatedGraphConv(hidden_dim, hidden_dim)

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim*self.window_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feature_dim)
        )

    def forward(self, data):
        """
        Forward pass of the RGGNN model.

        Parameters:
        - data (torch_geometric.data.Data): Input graph data.

        Returns:
        - x (torch.Tensor): Predicted node features.
        """
        x, edge_index = data.x, data.edge_index

        # Encoder
        x = self.encoder(x)

        # Processor
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Decoder
        x = torch.chunk(x, self.window_size, dim=0)
        x = torch.cat(x, dim=1)
        x = self.decoder(x)
        x = F.sigmoid(x)

        return x
