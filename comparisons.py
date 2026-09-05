import numpy as np
import matplotlib.pyplot as plt
import json
import logging
from pathlib import Path
from scipy.stats import pearsonr
from tqdm import tqdm
import seaborn as sns
import math



def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

# Return list of embeddings and ids
def load_embeddings(results_dir: str):
    embeddings = []
    ids = []
    
    # Go through every json file
    for f in Path(results_dir).glob('*.json'):
        with open(f, 'r') as file:
            # Store the embeddings and ids associated with them
            data = json.load(file)
            for job_id, content in data.items():
                embeddings.append(content['embedding'])
                ids.append(job_id)
    # Return the embeddings and ids
    return np.array(embeddings), ids


def create_heatmap(embeddings: np.ndarray, ids: list[str], output_dir: str):
    # To create the heatmap, a matrix is created of all the correlations
    differenceArray = []

    firstEmbeddingIndex = next((i for i, f in enumerate(ids) if f.startswith("job_B1.1.1_")), -1)
    
    # Go through each embedding
    for i, outerEmbedding in tqdm(enumerate(embeddings), desc="outerLoop", leave=False):
        """
        row = []
        # And get its correlation to all the other embeddings
        for j, innerEmbedding in tqdm(enumerate(embeddings), desc="innerLoop", leave=False):
            # The euclidean distance (commented out, but can be used if you comment out the pearson correlation)
            #row.append(np.sqrt(np.sum(np.square(innerEmbedding - outerEmbedding))))
            
            # The Pearson Correlation
            cor, p = pearsonr(innerEmbedding, outerEmbedding)
            row.append(cor)

        # Add this row of the correlations to the matrix
        differenceArray.append(row)
        """

        if i == firstEmbeddingIndex:
            continue

        row = []
        cor, p = pearsonr(embeddings[firstEmbeddingIndex], outerEmbedding)
        row.append(cor)

    return sum(row) / len(row)

        
    # # This is mainly for debugging purposes, and it does the same thing but prints out the correlations
    # # I don't just have this in the normal loop because it breaks the tqdm
    # for i, outerEmbedding in enumerate(embeddings):
    #     row = []
    #     for j, innerEmbedding in enumerate(embeddings):
    #         #row.append(np.sqrt(np.sum(np.square(innerEmbedding - outerEmbedding))))
    #         cor, p = pearsonr(innerEmbedding, outerEmbedding)
    #         row.append(cor * -1)
    #         print(f"i: {i}\t j: {j}\tcor: {cor}\tp: {p}")

    #     differenceArray.append(row)
    
    # Create either the heatmap or clustermap
    #im = sns.clustermap(differenceArray, cmap="YlGnBu", metric="correlation", tree_kws={"linewidths": 0.1})
    """
    im = sns.heatmap(differenceArray, cmap="YlGnBu")
    
    # Save the image
    output_path = Path(output_dir) / 'embedding_visualization.png'
    plt.savefig(output_path, dpi=1200, bbox_inches='tight')
    """


def generate_umap():
    logger = setup_logging()
    try:
        # This is the folder with all of the embeddings
            # This is where they are placed by default after running pipeline.py, however I often move them to a different folder so nothing gets overwritten
            # Also, if running on every chunk, then use data/embedding_results/chunks
        results_dir = "data/embedding_results"
        
        # This is where the image will be stored
        output_dir = "data/visualizations"
        Path(output_dir).mkdir(exist_ok=True)
        
        # Load embeddings
        embeddings, ids = load_embeddings(results_dir)
        
        # This is normally commented out, but if you want to just run it on a limited number of embeddings
        # embeddings = embeddings[:100]
        logger.info(f"Loaded {len(embeddings)} embeddings")
        
        # Generate visualization
        logger.info(create_heatmap(embeddings, ids, output_dir))
        logger.info("Generated heatmap")
        
        return f"Success"
        
    except Exception as e:
        logger.error(f"Error generating UMAP: {str(e)}")
        
        return f"Error: {str(e)}"

if __name__ == '__main__':
    res = generate_umap()

    print(res)