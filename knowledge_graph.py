# knowledge_graph.py

import networkx as nx
from pathlib import Path

def create_knowledge_graph(df_economy, entity_list, relation_list, entity_similarity_matrix, relation_similarity_matrix):
    # Initialize a directed graph
    G = nx.DiGraph()

    # Add nodes for entities
    for entity in entity_list:
        G.add_node(entity, label='entity')

    # Add nodes for relations
    for relation in relation_list:
        G.add_node(relation, label='relation')

    # Add edges for triplets with ideology labels and similarity values
    for index, row in df_economy.iterrows():
        ideology_label = row['cmp_code']  # Use cmp_code as ideology label
        triplets = row['relationships']

        for head, relation, tail in triplets:
            if head in G.nodes and tail in G.nodes and relation in G.nodes:
                # Add edge from head to tail with the relation as label
                G.add_edge(head, tail, label=relation, ideology=ideology_label)

    # Augment the graph with entity similarity values
    for i in range(len(entity_list)):
        for j in range(i + 1, len(entity_list)):
            entity_i = entity_list[i]
            entity_j = entity_list[j]
            similarity_value = entity_similarity_matrix[i][j]

            # Add or update undirected similarity edge between entities
            if similarity_value > 0.65:  # Optional: filter by threshold
                G.add_edge(entity_i, entity_j, label='similarity', weight=similarity_value)

    # Augment the graph with relation similarity values
    for i in range(len(relation_list)):
        for j in range(i + 1, len(relation_list)):
            relation_i = relation_list[i]
            relation_j = relation_list[j]
            similarity_value = relation_similarity_matrix[i][j]

            # Add or update undirected similarity edge between relations
            if similarity_value > 0.65:  # Optional: filter by threshold
                G.add_edge(relation_i, relation_j, label='similarity', weight=similarity_value)

    # Print the number of nodes and edges in the knowledge graph
    print(f"Knowledge Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Save the graph for further processing (e.g., visualization, GNN application)
    nx.write_gexf(G, str(Path(__file__).parent / 'knowledge_graph_with_similarity.gexf'))

    return G
