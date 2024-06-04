from torch import nn
import torch
import matplotlib.pyplot as plt
import networkx as nx


# Generic Feedforward Policy Network
class BinaryFeedForwardPolicyNetwork(nn.Module):
    def __init__(self):
        super(BinaryFeedForwardPolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2), nn.Softmax(dim=-1)
        )

    def forward(self, input_tensor, similarity):
        combined = torch.stack((input_tensor, similarity), 1)

        return self.layers(combined)

    def graph(self):
        # Use matplotlib and networkx to visualize the model
        # Create a directed graph
        G = nx.DiGraph()

        # Extract weights
        fc1_weights = self.layers[0].weight.detach().numpy()
        fc2_weights = self.layers[2].weight.detach().numpy()

        # Add nodes
        input_nodes = ["last_guess", "sim"]
        hidden_nodes = ["h1", "h2", "h3", "h4"]
        output_nodes = ["guess"]

        all_nodes = input_nodes + hidden_nodes + output_nodes
        G.add_nodes_from(all_nodes)

        # Add edges with weights
        for i, input_node in enumerate(input_nodes):
            for j, hidden_node in enumerate(hidden_nodes):
                G.add_edge(input_node, hidden_node, weight=fc1_weights[j, i])

        for i, hidden_node in enumerate(hidden_nodes):
            for j, output_node in enumerate(output_nodes):
                G.add_edge(hidden_node, output_node, weight=fc2_weights[j, i])

        # Define positions for a vertical layout
        pos = {}
        layer_dist = 2  # Distance between layers
        node_dist = 1.5  # Distance between nodes in the same layer

        # Position input layer nodes
        for idx, node in enumerate(input_nodes):
            pos[node] = (0, layer_dist * (len(input_nodes) - idx + 1))

        # Position hidden layer nodes
        for idx, node in enumerate(hidden_nodes):
            pos[node] = (layer_dist, layer_dist * (len(hidden_nodes) - idx))

        # Position output layer nodes
        for idx, node in enumerate(output_nodes):
            pos[node] = (2 * layer_dist, layer_dist * (len(output_nodes) - idx + 1.5))

        # Draw the graph
        plt.figure(figsize=(12, 8))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=2000,
            node_color="lightblue",
            font_size=10,
            font_weight="bold",
        )

        # Draw edge labels
        edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red", label_pos=0.7)

        plt.title("MLP Structure with Weights")
        plt.savefig("artifacts/plots/binary_ff_mlp.png")

        # Print weights
        print(self.state_dict())
