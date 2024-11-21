# embedding_generation.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def generate_embeddings(df_economy):
    # Initialize the Sentence Transformer model
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    # Extract unique entities (head + tail) and relations from the economy triplets
    entities = set()
    relations = set()

    for relationships in df_economy['relationships']:
        for triplet in relationships:
            head, relation, tail = triplet
            entities.update([head, tail])  # Add head and tail to entities
            relations.add(relation)  # Add relation to relations

    # Convert sets to lists for further processing
    entity_list = list(entities)
    relation_list = list(relations)

    # Generate embeddings for unique entities and relations
    entity_embeddings = model.encode(entity_list)
    relation_embeddings = model.encode(relation_list)

    # Calculate cosine similarity matrices for entities and relations
    entity_similarity_matrix = cosine_similarity(entity_embeddings)
    relation_similarity_matrix = cosine_similarity(relation_embeddings)

    # Print the entity similarity matrix
    print("Entity Similarity Matrix:")
    print(entity_similarity_matrix)

    # Print the relation similarity matrix
    print("\nRelation Similarity Matrix:")
    print(relation_similarity_matrix)

    # Optionally, print the entity names with their similarity scores (first 10)
    print("\nEntity Similarity (first 10):")
    for i in range(min(10, len(entity_list))):
        print(f"{entity_list[i]}: {entity_similarity_matrix[i]}")

    # Optionally, print the relation names with their similarity scores (first 10)
    print("\nRelation Similarity (first 10):")
    for i in range(min(10, len(relation_list))):
        print(f"{relation_list[i]}: {relation_similarity_matrix[i]}")

    return entity_list, relation_list, entity_embeddings, relation_embeddings, entity_similarity_matrix, relation_similarity_matrix
