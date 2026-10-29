import numpy as np
import json
from pathlib import Path
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

"""
Things to do:
- Improve line visualization function appearance?
- Fully implement chunk heatmap visualization classification accuracy
- Save the model and reload it for testing
- Try for better accuracy with different parameters
- Get working on the chunk-level classification (Maybe the answer is to only train on text-level embeddings, and then use the chunk-level embeddings for testing)
- Currently, it looks like the model is either overfitting or the data is set up in a way that makes classification too easy but innacurate. Need to check the data and make sure that the embeddings are paired correctly and that the training and testing sets are not too similar.
"""

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

config: dict = {
    "author": "Emma",
    "input_dim": 1024,
    "hidden_dim": None,
    "optimizer": "Adam",
    "weight_decay": None,
    "dropout_rate": 0.4,
    "num_epochs": 30,
    "batch_size": 32,
    "test_split": 0.5,
    "aelfric_threshold": 0.2,
    "learning_rate": 1e-3,
    "random_seed": 11,
    "num_samples": int(1e4),
    "aelfric_path": "data/sorted_embeddings/Train/Large Chunk/Aelfric",
    "unknown_path": "data/sorted_embeddings/Train/Large Chunk/Unknown"
}

class LinearBinaryModel(nn.Sequential):
    def __init__(self):
        layers = OrderedDict()
        prev_dim = config["input_dim"]
        
        if "hidden_dim" in config:
            if config["hidden_dim"] is not None:
                name = "hidden"
                layers[name] = nn.Linear(prev_dim, config["hidden_dim"])
                layers[f"relu1"] = nn.ReLU()
                layers[f"dropout1"] = nn.Dropout(config["dropout_rate"])
                layers["output"] = nn.Linear(config["hidden_dim"], 1)
            else:
                layers["output"] = nn.Linear(prev_dim, 1)
        super().__init__(layers)

'''
class Dataset(Dataset):
    def __init__(self, data, labels, ids):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.ids = ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        ids = self.ids[idx]
        return x, y, ids
'''

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)
    

"""
Creates a visualization of the training and testing errors over epochs. 
The function dynamically adjusts the figure size based on the number of lines in the model description to ensure that all text is visible. 
It plots the error lines for different classes and saves the resulting figure as a PNG file.

model: The trained model whose description will be displayed below the plot.
lines: A dictionary where keys are labels (e.g., "In Sample Error") and values are lists of error values over epochs.
title: The title of the plot.
filename: The filename for saving the plot (without extension).
other_data: Additional text data to display below the plot (e.g., hyperparameters or notes).
"""
def visualization(model, lines : dict, title: str, filename: str, other_data: str) -> None:
    model_string = str(model)

    # Count lines to estimate height needed 
    model_lines = len(model_string.splitlines())
    extra_height = 0.1 + model_lines * 0.015  # adjust for text length
    
    # Create figure with dynamic bottom margin
    fig, ax = plt.subplots(figsize=(15, 5 + extra_height * 10))
    fig.subplots_adjust(bottom=(extra_height * 10) / ((extra_height * 10) + 5))

    colors = ["#e41a1c", "#377eb8", "#4daf4a"]

    for i, (label, line) in enumerate(lines.items()):
        ax.plot(line, color=colors[i], label=label, alpha=0.7)
    
    ax.set_ylim(-1, 31)
    ax.set_yticks(range(0, 35, 10))
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")

    # Add model description below the plot
    fig.text(
        0.1, 0.02, model_string,
        fontsize=10,
        va='bottom',
        ha='left',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    )
    # Add model description below the plot
    fig.text(
        0.5, 0.02, other_data,
        fontsize=10,
        va='bottom',
        ha='left',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    )

    plt.savefig(f"data/visualizations/{filename}.png")  

"""
Returns list of embeddings and ids

This function loads embeddings from JSON files in the specified directory. 
It collects embeddings and their associated IDs, while also keeping track of unique text IDs. 
The function limits the number of chunks processed to 250 for efficiency.

results_dir: Directory containing the JSON files with embeddings.
max_num_of_texts: Maximum number of embeddings to process (default is 250).
Returns: A tuple containing a NumPy array of embeddings and a list of unique text IDs.
"""
def load_embeddings(results_dir: str, max_num_of_texts: int = 500):
    embeddings: list = []
    ids: list = []
    num_of_texts = 0
    
    # Go through every json file in the directory
    for f in tqdm(Path(results_dir).glob('*.json')):
        with open(f, 'r') as file:
            # Store the embeddings and ids associated with them
            data = json.load(file)

            job_id = next(iter(data))  # Get the job_id (Same as list(data.keys())[0])
            embeddings.append(data[job_id]['embedding']) # Appends text embedding
            text_id = job_id.split('_')[1] # Extracts text ID from job_id
            ids.append(text_id) # Appends text ID to the list of IDs

        num_of_texts += 1
        if num_of_texts > max_num_of_texts:
            break
                
    # Return the embeddings and ids
    return np.array(embeddings)

"""
THIS ONLY WORKS ON TEXT LEVEL, NOT CHUNK LEVEL. NEED TO FIX THIS TO KEEP CHUNKS TOGETHER.

Loads Aelfric and Unknown embeddings and splits them into training and testing sets

a_path: Path to the directory containing Aelfric embeddings.
u_path: Path to the directory containing Unknown embeddings.
test_split: The fraction of the dataset to include in the testing set.
random_seed: The seed for random number generation.
Returns: A tuple containing training embeddings, testing embeddings, training labels, testing labels, training IDs, testing IDs, and Aelfric embeddings.
"""
def load_data(config: dict) -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[str], list[str], np.ndarray]:
    a_embeddings = np.array([]) # np.array of Aelfric embeddings
    a_ids: list[str] = [] # List of Aelfric IDs
    
    u_embeddings = np.array([]) # np.array of Unknown embeddings
    u_ids: list[str] = [] # List of Unknown IDs

    embeddings = np.array([]) # np.array of all embeddings
    y: list[int] = [] # List of all labels (+1 for Aelfric, 0 for Unknown)
    ids: list[str] = [] # List of all IDs

    # Get the Aelfric embeddings
    a_embeddings, a_ids = load_embeddings(config["aelfric_path"], config["num_samples"])

    # Get the Unknown embeddings
    u_embeddings, u_ids = load_embeddings(config["unknown_path"], config["num_samples"])

    # Create the list of all the embeddings, ids, and labels
    embeddings = np.concatenate((a_embeddings, u_embeddings), axis=0) 
    y = [1] * len(a_ids) + [0] * len(u_ids)
    ids = a_ids + u_ids

    # Randomly split the full works while keeping embeddings, IDs, and labels aligned
    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(embeddings, y, ids, test_size=config["test_split"], stratify=y)

    # Save the Aelfric training embeddings for later use in predictions
  

    return x_train, x_test, y_train, y_test, id_train, id_test


"""
Create Aelfric-Aelfric (positive) and Aelfric-Unknown (zero) pairs.

embs: List of embeddings (np.ndarray).
labels: List of corresponding labels (1 for Aelfric, 0 for Unknown).
num_samples: The number of samples to draw from each class.
Returns: Two lists of concatenated embeddings.
"""
def pair_embeddings(embs: list[np.ndarray], labels: list[int], num_samples:int = 250) -> tuple[list[np.ndarray], list[np.ndarray]]:
    a_embs = []
    u_embs = []

    for i, emb in enumerate(embs):
        if labels[i] == 1:
            a_embs.append(emb)
        else:
            u_embs.append(emb)

    random.seed(config["random_seed"])  # Ensure reproducibility for sampling
    a_sample = random.sample(a_embs, k=min(num_samples, len(a_embs))) ### Eventually the samples need to be controlled better
    random.seed(config["random_seed"])  # Ensure reproducibility for sampling
    u_sample = random.sample(u_embs, k=min(num_samples, len(u_embs)))

    aa_pairs = []
    au_pairs = []

    # Aelfric–Aelfric pairs (1)
    for emb_a in a_sample:
        for emb_b in a_sample:
            aa_pairs.append(np.concatenate([emb_a, emb_b]))

    # Aelfric–Unknown pairs (0)
    for emb_a in a_sample:
        for emb_b in u_sample:
            au_pairs.append(np.concatenate([emb_a, emb_b]))

    # Unknown-Aelfric pairs (0)
    for emb_a in u_sample:
        for emb_b in a_sample:
            au_pairs.append(np.concatenate([emb_a, emb_b]))

    random.seed(config["random_seed"])  # Ensure reproducibility for sampling
    au_sample = random.sample(au_pairs, k=min(num_samples*3, len(au_pairs)))  # Sample to limit size

    return aa_pairs, au_sample


"""
Loads the data and prepares training pairs

Returns: Training and testing sets along with Aelfric embeddings for later use in predictions.
"""
def prepare_data():
    # Get aelfric and unknown embeddings
    aelfric_dict = load_embeddings("data/sorted_embeddings/Train/Large Chunk/Aelfric")
    unknown_dict = load_embeddings("data/sorted_embeddings/Train/Large Chunk/Unknown")

    # Stack the embeddings into matrices
    X_a = aelfric_dict # shape (N_a, 1024)
    X_u = unknown_dict # shape (N_u, 1024)

    y_a = np.ones(len(X_a)) # +1 for Aelfric
    y_u = np.zeros(len(X_u)) # 0 for Unknown

    X = np.vstack([X_a, X_u]) # shape (N_total, 1024)
    y = np.concatenate([y_a, y_u]) # shape (N_total,)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


"""
Creates the model, trains it on the training data, and evaluates it on both training and testing data.

x_train: Training embeddings (np.ndarray).
y_train: Training labels (np.ndarray).
x_test: Testing embeddings (np.ndarray).
y_test: Testing labels (np.ndarray).
a_train_embeddings: Aelfric training embeddings (np.ndarray).
num_epochs: The number of epochs to train the model.
batch_size: The batch size for training.
Returns: The trained model along with dictionaries containing in-sample and out-of-sample accuracies for overall, Aelfric, and Unknown classes.
"""
def train_linear_model(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, config: dict) -> tuple[LinearBinaryModel, dict, dict]:
    x_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)  # shape (N_total, 1)

    # Create per-sample weights: slightly downweight Unknown (-1)
    #weights = np.ones(len(y_train), dtype=np.float32)
    #weights[y_train == -1] = 0.9
    #weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)

    #model = LinearBinaryModel(input_dim=config["input_dim"], hidden_dims=config["hidden_dims"], dropout_rates=config["dropout_rates"]).to(device)
    model = LinearBinaryModel()
    

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Dataset with weights
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    in_acc_total = []
    in_acc_aelfric = []
    in_acc_unknown = []
    out_acc_total = []
    out_acc_aelfric = []
    out_acc_unknown = []

    for epoch in tqdm(range(config["num_epochs"]), desc="Training Epochs"):

        # Train for one epoch
        model.train()

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)  # predicted score
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        # Turns off dropout for evaluation
        model.eval()

        # After each epoch, save in-sample and out-of-sample accuracy
        in_predictions = predict_linear_model(model, x_train)
        in_acc_total.append(accuracy(y_train, in_predictions)*100)
        in_accs = class_accuracy(y_train, in_predictions)
        in_acc_aelfric.append(in_accs["Aelfric"]*100)
        in_acc_unknown.append(in_accs["Unknown"]*100)

        out_predictions = predict_linear_model(model, x_test)
        out_acc_total.append(accuracy(y_test, out_predictions)*100)
        out_accs = class_accuracy(y_test, out_predictions)
        out_acc_aelfric.append(out_accs["Aelfric"]*100)
        out_acc_unknown.append(out_accs["Unknown"]*100)

    return model, \
           {"Accuracy": in_acc_total, "Aelfric Accuracy": in_acc_aelfric, "Unknown Accuracy": in_acc_unknown}, \
           {"Accuracy": out_acc_total, "Aelfric Accuracy": out_acc_aelfric, "Unknown Accuracy": out_acc_unknown}


"""
Predicts the class (+1 for Aelfric, 0 for Unknown) for new embeddings using the trained model.
If Aelfric embeddings are provided, it pairs each new embedding with all Aelfric embeddings to make predictions based on the average score.

model: The trained LinearBinaryModel.
x_new: New embeddings to classify (np.ndarray of shape (N, 1024) or (1024,)).
a_train_embeddings: Optional Aelfric embeddings to pair with new embeddings for prediction.
Returns: A list of predicted classes (+1 for Aelfric, 0 for Unknown) for each new embedding.
"""
def predict_linear_model(model, x_new):
    """
    x_new: np.array of shape (1024,) or (N, 1024)
    returns: list of +1 (Aelfric) or 0 (Unknown)
    """
    if x_new.ndim == 1:
        x_new = x_new[np.newaxis, :]  # shape (1, 2048)
    
    with torch.no_grad():
        X_tensor = torch.tensor(x_new, dtype=torch.float32)
        probs = torch.sigmoid(model(X_tensor)).reshape(-1).cpu().numpy()
        return [1 if p > 0.5 else 0 for p in probs]
    
"""
Predicts the class (+1 for Aelfric, 0 for Unknown) for a single embedding by pairing it with a sample of Aelfric embeddings and averaging the model's scores.
(MIGHT CHANGE HOW WE AVERAGE SCORES INSTEAD OF JUST TAKING THE MEAN, MAYBE WEIGHT BY CONFIDENCE OR SOMETHING)

model: The trained LinearBinaryModel.
test_emb: A single embedding to classify (np.ndarray of shape (1024,)).
a_train_embeddings: A list of Aelfric training embeddings to pair with the test embedding for prediction.
Returns: A single predicted class (+1 for Aelfric, 0 for Unknown) for the test embedding based on the average score from the model.
"""
def predict_single_embedding(model, test_emb: np.ndarray, a_train_embeddings: list[np.ndarray]) -> int:
    # Sample fewer Aelfric embeddings for speed
    random.seed(config["random_seed"])  # Ensure reproducibility for sampling
    aelfrics = random.sample(a_train_embeddings.tolist(), k=min(len(a_train_embeddings), 100))

    pairs = [np.concatenate([a, test_emb]) for a in aelfrics]  # shape (N_a, 2048)
    x_tensor = torch.tensor(np.stack(pairs), dtype=torch.float32).to(device)

    # Switches to eval
    model.eval()

    with torch.no_grad():
        logits = model(x_tensor).squeeze()  # one score per pair
        probs = torch.sigmoid(logits).cpu().numpy()  # convert to probabilities

    # Aggregate:
    avg_score = np.mean(probs)  # average probability across all pairs
    return 1 if avg_score > 0.5 else 0


def log_test_data(model, x_train, y_train, x_test, y_test, a_train_embeddings, y_pred):
    model.eval()

    aelfric_scores = []
    unknown_scores = []

    for test_emb, true_label in zip(x_test, y_test):
        # Pair this test embedding with Aelfric reference embeddings
        random.seed(config["random_seed"])  # Ensure reproducibility for sampling
        aelfrics = random.sample(
            a_train_embeddings.tolist(),
            k=min(len(a_train_embeddings), 100)
        )

        pairs = [np.concatenate([a, test_emb]) for a in aelfrics]
        x_tensor = torch.tensor(
            np.stack(pairs),
            dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            logits = model(x_tensor).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()

        avg_score = np.mean(probs)

        if true_label == 1:
            aelfric_scores.append(avg_score)
        else:
            unknown_scores.append(avg_score)

    print("\n===== RAW SCORE DISTRIBUTION =====")

    print("\nAelfric:")
    print(f"  Mean: {np.mean(aelfric_scores):.4f}")
    print(f"  Min:  {np.min(aelfric_scores):.4f}")
    print(f"  Max:  {np.max(aelfric_scores):.4f}")
    print(f"  Std:  {np.std(aelfric_scores):.4f}")

    print("\nUnknown:")
    print(f"  Mean: {np.mean(unknown_scores):.4f}")
    print(f"  Min:  {np.min(unknown_scores):.4f}")
    print(f"  Max:  {np.max(unknown_scores):.4f}")
    print(f"  Std:  {np.std(unknown_scores):.4f}")

    print("\n===== DATA DISTRIBUTION =====")

    print("Train:")
    print("  Aelfric:", np.sum(y_train == 1))
    print("  Unknown:", np.sum(y_train == 0))
    print("Train embeddings shape:", x_train.shape)
    print(x_train)
    print(y_train)

    print("Test:")
    print("  Aelfric:", np.sum(y_test == 1))
    print("  Unknown:", np.sum(y_test == 0))

    print("Aelfric training embeddings:", len(a_train_embeddings))
    print("Aelfric test embeddings:", np.sum(y_test == 1))
    print("Unknown test embeddings:", np.sum(y_test == 0))
    print("Aelfric training embeddings shape:", a_train_embeddings.shape)
    print(a_train_embeddings)

    print("\n===== PREDICTIONS =====\n")

    print("Predicted Aelfric:", np.sum(np.array(y_pred) == 1))
    print("Predicted Unknown:", np.sum(np.array(y_pred) == 0))

    print("Actual Aelfric:", np.sum(np.array(y_test) == 1))
    print("Actual Unknown:", np.sum(np.array(y_test) == 0))

    """
    print("\nIndividual scores:")

    print("\nAelfric:")
    for index, score in enumerate(aelfric_scores):
        if index < 30:  # Print only the first few scores
            print(f"  {score:.4f}")

    print("\nUnknown:")
    for index, score in enumerate(unknown_scores):
        if index < 30:  # Print only the first few scores
            print(f"  {score:.4f}")
    """


"""
Calculates overall accuracy by comparing true labels with predicted labels.

y_true: List of true labels (+1 for Aelfric, 0 for Unknown).
y_pred: List of predicted labels (+1 for Aelfric, 0 for Unknown).
Returns: The overall accuracy as a float (between 0 and 1).
"""
def accuracy(y_true: list[int], y_pred: list[int]) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


"""
Calculates class-specific accuracy for Aelfric and Unknown classes by comparing true labels with predicted labels.

y_true: List of true labels (+1 for Aelfric, 0 for Unknown).
y_pred: List of predicted labels (+1 for Aelfric, 0 for Unknown).
Returns: A dictionary containing the accuracy for each class, with keys "Aelfric" and "Unknown".
"""
def class_accuracy(y_true: list[int], y_pred: list[int]) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Check if class has any samples before computing mean
    aelfric_mask = y_true == 1
    if np.any(aelfric_mask):
        acc_aelfric = np.mean(y_pred[aelfric_mask] == 1)
    else:
        acc_aelfric = np.nan
    
    unknown_mask = y_true == 0
    if np.any(unknown_mask):
        acc_unknown = np.mean(y_pred[unknown_mask] == 0)
    else:
        acc_unknown = np.nan
    
    return {"Aelfric": acc_aelfric, "Unknown": acc_unknown}

"""
Calculates error based on accuracy dictionaries for in-sample and out-of-sample data.

accs: A dictionary containing accuracy values for overall, Aelfric, and Unknown classes.
Returns: A dictionary containing error values for overall, Aelfric, and Unknown classes, calculated as 100 minus the corresponding accuracy values.
"""
def error(accs : dict) -> dict: 
    errors = {}
    errors["Error"] = 100 - np.array(accs["Accuracy"])
    errors["Aelfric Error"] = 100 - np.array(accs["Aelfric Accuracy"])
    errors["Unknown Error"] = 100 - np.array(accs["Unknown Accuracy"])
    return errors

"""
Save the config and model to a file
"""
def save_model(config, model, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config
    }, path)
    print(f"Model saved to {path}") 


"""
Load the config and model from a file
"""
def load_model(path):
    global config
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = LinearBinaryModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(f"Loaded model from {path}")
    return model
    

def main():


    logger = setup_logging()

    model_loaded = True  # Set to True if you want to load a pre-trained model instead of training a new one

    if model_loaded:
        model_path = "models/linear_model (92_67_100) (10_28_2026 18-57).pth"  # Update with your model path
        #model_path = "models/linear_model E(95_81_99) (10_12_2026 16-52).pth"
        model = load_model(model_path)

        # Prepare the training and testing data
        x_train, x_test, y_train, y_test = prepare_data()   

        # Find and print accuracies
        y_pred = predict_linear_model(model, x_test)

        acc_overall = accuracy(y_test, y_pred)
        class_acc = class_accuracy(y_test, y_pred)

        #log_test_data(model, x_train, y_train, x_test, y_test, a_train_embeddings, y_pred, random_seed=config["random_seed"])

        logger.info(f"Overall test accuracy: {acc_overall*100:.2f}%")
        logger.info(f"Aelfric accuracy: {class_acc['Aelfric']*100:.2f}%")
        logger.info(f"Unknown accuracy: {class_acc['Unknown']*100:.2f}%")   


    else:

        # Train the model
        x_train, x_test, y_train, y_test = prepare_data()    

        model, in_accs, out_accs = train_linear_model(x_train, y_train, x_test, y_test, config)
        logger.info("Training complete.")

        # Find and print accuracies
        y_pred = predict_linear_model(model, x_test)

        acc_overall = accuracy(y_test, y_pred)
        class_acc = class_accuracy(y_test, y_pred)

        #log_test_data(model, x_train, y_train, x_test, y_test, y_pred, random_seed=config["random_seed"])

        logger.info(f"Overall test accuracy: {acc_overall*100:.2f}%")
        logger.info(f"Aelfric accuracy: {class_acc['Aelfric']*100:.2f}%")
        logger.info(f"Unknown accuracy: {class_acc['Unknown']*100:.2f}%")

        # Save the model with descriptive filename
        acc_str = f"{round(acc_overall*100)}_{round(class_acc['Aelfric']*100)}_{round(class_acc['Unknown']*100)}"
        timestamp_str = datetime.now().strftime("%m_%d_%Y %H-%M")
        model_filename = f"models/linear_model ({acc_str}) ({timestamp_str}).pth" 
        #I put an E to mean that it's mine, but actually we should probably have a number representing the number of hidden layers probably? Doesn't matter as much as I thought it did though because the code is flexible now.
        save_model(config, model, model_filename)

        # Visualizations
        in_errors = error(in_accs)
        out_errors = error(out_accs)

        visualization(model, in_errors, "In Sample Error", "inSample", "")
        visualization(model, out_errors, "Out of Sample Error", "outSample", "")
        visualization(model, {"In Sample Error": in_errors["Error"], "Out of Sample Error": out_errors["Error"]}, "Both Error", "bothSample", "")


if __name__ == '__main__':
    #res = generate_umap()

    #print(res)

    main()