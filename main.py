# main.py

from data_preprocessing import load_and_preprocess_data
from embedding_generation import generate_embeddings
from knowledge_graph import create_knowledge_graph
from model_training import train_gat_model
from evaluation import evaluate_model
from dbpedia_enrichment import enrich_graph_with_dbpedia
import networkx as nx
from torch_geometric.utils import from_networkx

def main():
    # Step 1: Load and preprocess data
    df_economy = load_and_preprocess_data()

    # Step 2: Generate embeddings
    entity_list, relation_list, entity_embeddings, relation_embeddings, entity_similarity_matrix, relation_similarity_matrix = generate_embeddings(df_economy)

    # Step 3: Create knowledge graph
    G = create_knowledge_graph(df_economy, entity_list, relation_list, entity_similarity_matrix, relation_similarity_matrix)

    # Step 4: Convert graph to PyTorch Geometric Data object
    from convert_graph_to_pyg import convert_graph_to_pyg
    pyg_graph = convert_graph_to_pyg(G, entity_list, relation_list, entity_embeddings, relation_embeddings, df_economy)

    # Step 5: Train GAT model
    model = train_gat_model(pyg_graph, df_economy)

    # Step 6: Evaluate model
    accuracy = evaluate_model(model, pyg_graph)
    print(f"Model Accuracy: {accuracy}")

    # Step 7: Enrich graph with DBpedia (if needed)
    # For the graph without edges
    G_no_edges = nx.DiGraph()
    # Add nodes for entities and relations
    for entity in entity_list:
        G_no_edges.add_node(entity, label='entity')
    for relation in relation_list:
        G_no_edges.add_node(relation, label='relation')
    pyg_graph_no_edges = from_networkx(G_no_edges)
    # Assign features and labels
    pyg_graph_no_edges.x = pyg_graph.x
    pyg_graph_no_edges.y = pyg_graph.y

    # Enrich the graph
    pyg_graph_enriched_no_edges = enrich_graph_with_dbpedia(pyg_graph_no_edges)

    # You can proceed to train and evaluate a model on the enriched graph if needed

if __name__ == "__main__":
    main()
