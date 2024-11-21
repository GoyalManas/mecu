# dbpedia_enrichment.py

from SPARQLWrapper import SPARQLWrapper, JSON
import urllib.parse
from torch_geometric.utils import to_networkx, from_networkx
import torch

def enrich_graph_with_dbpedia(pyg_graph_no_edges):
    # Function to generate DBpedia URI from entity name
    def generate_dbpedia_uri(entity_name):
        base_uri = "http://dbpedia.org/resource/"
        entity_uri = base_uri + urllib.parse.quote(str(entity_name).replace(" ", "_"))
        return entity_uri

    # Function to query DBpedia for additional relationships
    def query_dbpedia(entity_uri):
        sparql = SPARQLWrapper("https://dbpedia.org/sparql")
        query = f"""
        SELECT ?property ?value
        WHERE {{
            <{entity_uri}> ?property ?value
        }} LIMIT 100
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            return results["results"]["bindings"]
        except Exception as e:
            print(f"Error querying DBpedia for {entity_uri}: {e}")
            return []

    # Enrich PyG graph using DBpedia
    def enrich_pyg_graph_no_edges(pyg_graph):
        # Convert PyG graph to NetworkX for easier manipulation
        nx_graph = to_networkx(pyg_graph, to_undirected=True)
        # Create a static list of nodes to iterate over
        nodes = list(nx_graph.nodes)
        # Add edges based on DBpedia
        for node in nodes:
            entity_uri = generate_dbpedia_uri(node)
            dbpedia_data = query_dbpedia(entity_uri)
            for result in dbpedia_data:
                property_uri = result["property"]["value"]
                value = result["value"]["value"]
                if result["value"]["type"] == "uri":
                    # Add new edge with property as label
                    nx_graph.add_edge(node, value, label=property_uri)
                else:
                    # Add new node attribute if the value is not a URI
                    if node in nx_graph.nodes:
                        nx_graph.nodes[node][property_uri] = value
        # Convert back to PyTorch Geometric
        enriched_pyg_graph = from_networkx(nx_graph)
        # Add original node features and labels
        enriched_pyg_graph.x = pyg_graph.x  # Preserve original features
        enriched_pyg_graph.y = pyg_graph.y  # Preserve original labels
        return enriched_pyg_graph

    # Enrich the graph
    pyg_graph_enriched_no_edges = enrich_pyg_graph_no_edges(pyg_graph_no_edges)

    # Print details of the enriched graph
    print(f"Enriched PyG graph: {pyg_graph_enriched_no_edges.num_nodes} nodes, {pyg_graph_enriched_no_edges.num_edges} edges.")

    # Save the enriched graph for future use
    torch.save(pyg_graph_enriched_no_edges, str(Path(__file__).parent / 'pyg_graph_enriched_no_edges.pt'))
    print("Enriched PyG graph saved as 'pyg_graph_enriched_no_edges.pt'.")

    return pyg_graph_enriched_no_edges
