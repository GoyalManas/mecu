# convert_graph_to_pyg.py

import torch
from torch_geometric.utils import from_networkx

def convert_graph_to_pyg(G, entity_list, relation_list, entity_embeddings, relation_embeddings, df_economy):
    # Ensure all edges have the same attributes
    for u, v, data in G.edges(data=True):
        # Set default values for missing attributes
        if 'label' not in data:
            data['label'] = 'none'  # or some default value
        if 'ideology' not in data:
            data['ideology'] = 'unknown'  # or some default value
        if 'weight' not in data:
            data['weight'] = 1.0  # Set a default similarity weight for edges that lack it

    # Convert the NetworkX graph to a PyTorch Geometric Data object
    pyg_graph = from_networkx(G)

    # Create feature tensors for the nodes (entity and relation embeddings)
    num_entities = len(entity_list)
    num_relations = len(relation_list)

    # Create node features for entities and relations
    node_features = torch.zeros((G.number_of_nodes(), entity_embeddings.shape[1]))

    # Assign embeddings to nodes based on whether they are entities or relations
    for i, entity in enumerate(entity_list):
        node_idx = list(G.nodes).index(entity)  # Get node index in the graph
        node_features[node_idx] = torch.tensor(entity_embeddings[i])

    for i, relation in enumerate(relation_list):
        node_idx = list(G.nodes).index(relation)  # Get node index in the graph
        node_features[node_idx] = torch.tensor(relation_embeddings[i])

    # Assign node features to the PyTorch Geometric Data object
    pyg_graph.x = node_features

    # For node labels (ideology labels for entities, dummy labels for relations)
    node_labels = torch.zeros(G.number_of_nodes(), dtype=torch.long)

    for entity in entity_list:
        node_idx = list(G.nodes).index(entity)
        ideology_label = df_economy[df_economy['relationships'].apply(lambda x: entity in [t[0] for t in x])]['cmp_code'].iloc[0]
        node_labels[node_idx] = int(ideology_label)  # Convert ideology label to integer

    pyg_graph.y = node_labels  # Set labels to PyG graph

    # Edge attributes (for similarity)
    edge_weights = []
    for edge in G.edges(data=True):
        edge_weights.append(edge[2].get('weight', 1.0))  # Use weight if available, else 1.0

    pyg_graph.edge_attr = torch.tensor(edge_weights, dtype=torch.float)

    return pyg_graph
