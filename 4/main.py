import numpy as np

def cosine_similarity(vector_a, vector_b):
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
    vector_a (list or np.array): First vector.
    vector_b (list or np.array): Second vector.

    Returns:
    float: Cosine similarity between vector_a and vector_b.
    """
    # Convert lists to numpy arrays if they aren't already
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    
    # Compute the dot product of the two vectors
    dot_product = np.dot(vector_a, vector_b)
    
    # Compute the norm (magnitude) of each vector
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    # Calculate the cosine similarity
    cosine_sim = dot_product / (norm_a * norm_b)
    
    return cosine_sim

# Example usage
vector1 = [1, 2, 3]
vector2 = [4, 5, 6]
print(f"Cosine similarity between {vector1} and {vector2}: {cosine_similarity(vector1, vector2)}")

vector3 = [1, 0, 0]
vector4 = [0, 1, 0]
print(f"Cosine similarity between {vector3} and {vector4}: {cosine_similarity(vector3, vector4)}")

vector5 = [1, 1, 1]
vector6 = [2, 2, 2]
print(f"Cosine similarity between {vector5} and {vector6}: {cosine_similarity(vector5, vector6)}")

vector7 = [1, 0, -1]
vector8 = [-1, 0, 1]
print(f"Cosine similarity between {vector7} and {vector8}: {cosine_similarity(vector7, vector8)}")