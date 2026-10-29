import numpy as np
import json
from pathlib import Path
import logging
import torch
import torch.nn as nn
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials


logging.basicConfig(level=logging.INFO, format="%(message)s")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

model_name : str = "models/linear_model (94._91._94.) (10_21_2026 22-26).pth"
model_loaded = True  # Set to True if you want to load a pre-trained model instead of training a new one

# Defines the configuration for the model and training parameters
config: dict = {
    "author": "Mark",
    "input_dim": 2048,
    "hidden_dim": 500,
    "optimizer": "AdamW",
    "weight_decay": 0.35,
    "dropout_rate": 0.3,
    "num_epochs": 75,
    "batch_size": 32,
    "test_split": 0.3,
    "threshold": 0.2,
    "learning_rate": 4e-4,
    "random_seed": 11,
    "num_samples": int(1e4),
    "aelfric_path": "data/sorted_embeddings/Train/Large Chunk/Aelfric",
    "unknown_path": "data/sorted_embeddings/Train/Large Chunk/Unknown"
}

accuracy_data: dict = {}


class LinearBinaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        if config["hidden_dim"] is not None:
            self.hidden = nn.Linear(config["input_dim"], config["hidden_dim"])  # input → hidden
            self.relu = nn.ReLU()                            # non-linearity
            self.dropout = nn.Dropout(config["dropout_rate"])
            self.output = nn.Linear(config["hidden_dim"], 1)          # hidden → output
        else:
            self.output = nn.Linear(config["input_dim"], 1)          # input → output

    def forward(self, x):
        if config["hidden_dim"] is not None:
            x = self.hidden(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.output(x)   # scalar score

        return x


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
def visualization(model: LinearBinaryModel, lines : dict, title: str, filename: str, other_data: str) -> None:
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
def load_embeddings(results_dir: str, max_num_of_texts: int = 500) -> tuple[np.ndarray, list]:
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
    return np.array(embeddings), ids

"""
THIS ONLY WORKS ON TEXT LEVEL, NOT CHUNK LEVEL. NEED TO FIX THIS TO KEEP CHUNKS TOGETHER.

Loads Aelfric and Unknown embeddings and splits them into training and testing sets

a_path: Path to the directory containing Aelfric embeddings.
u_path: Path to the directory containing Unknown embeddings.
test_split: The fraction of the dataset to include in the testing set.
random_seed: The seed for random number generation.
Returns: A tuple containing training embeddings, testing embeddings, training labels, testing labels, training IDs, testing IDs, and Aelfric embeddings.
"""
def load_data() -> tuple[np.ndarray, np.ndarray, list[int], list[int], list[str], list[str], np.ndarray]:
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
    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(embeddings, y, ids, test_size=config["test_split"], random_state=config["random_seed"], stratify=y)

    # Save the Aelfric training embeddings for later use in predictions
    a_train_embeddings = x_train[np.array(y_train) == 1]  # Select embeddings where label is 1 (Aelfric)

    return x_train, x_test, y_train, y_test, id_train, id_test, a_train_embeddings


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
def prepare_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train_old, x_test, y_train_old, y_test, _, _, a_train_embeddings = load_data() #"data/sorted_embeddings/Aelfric", "data/sorted_embeddings/Unknown", 0.5, 13)
    
    logging.info("Loaded embeddings...")
    logging.debug(f"Train Embeddings: {x_train_old[:5]} ({len(x_train_old)})")
    logging.debug(f"Train Labels: {y_train_old[:5]} ({len(y_train_old)})")
    logging.debug(f"Test Embeddings: {x_test[:5]} ({len(x_test)})")
    logging.debug(f"Test Labels: {y_test[:5]} ({len(y_test)})")

    aa_pairs, au_pairs = pair_embeddings(x_train_old, y_train_old, num_samples=config["num_samples"])

    # --- Combine and label training pairs ---
    x_aa = np.stack(aa_pairs)
    x_au = np.stack(au_pairs)

    y_aa = np.ones(len(x_aa))   # +1 for Aelfric–Aelfric
    y_au = np.zeros(len(x_au))  # 0 for Aelfric–Unknown

    x_train = np.vstack([x_aa, x_au])
    y_train = np.concatenate([y_aa, y_au])

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), a_train_embeddings


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
def train_linear_model(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, a_train_embeddings: np.ndarray) -> tuple[LinearBinaryModel, dict, dict]:
    x_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)  # shape (N_total, 1)

    # Create per-sample weights: slightly downweight Unknown (-1)
    #weights = np.ones(len(y_train), dtype=np.float32)
    #weights[y_train == -1] = 0.9
    #weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)

    model = LinearBinaryModel().to(device)

    #This is added to penalize mistakes on Aelfric labels by weighing them heavier
    pos_weight = torch.tensor([np.sum(y_train == 0) / np.sum(y_train == 1)], dtype=torch.float32).to(device)  # Unknown/Aelfric ratio
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Set up the optimizer based on the configuration
    if config["weight_decay"] and config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["weight_decay"] and config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["weight_decay"] is None and config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    elif config["weight_decay"] is None and config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    else:
        logging.warning("Optimizer or weight_decay not specified correctly. Using default Adam optimizer without weight decay.")
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
        in_predictions = predict_linear_model(model, x_train, a_train_embeddings=None)
        in_acc_total.append(accuracy(y_train, in_predictions)*100)
        in_accs = class_accuracy(y_train, in_predictions)
        in_acc_aelfric.append(in_accs["Aelfric"]*100)
        in_acc_unknown.append(in_accs["Unknown"]*100)

        out_predictions = predict_linear_model(model, x_test, a_train_embeddings)
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
def predict_linear_model(model: LinearBinaryModel, x_new: np.ndarray, a_train_embeddings: np.ndarray = None) -> list[int]:
    if a_train_embeddings is None:
        # Standard case (already 2048-dim)
        if x_new.ndim == 1:
            x_new = x_new[np.newaxis, :]

        # Double checks that we're not training
        model.eval()

        with torch.no_grad():
            x_tensor = torch.tensor(x_new, dtype=torch.float32).to(device)
            logits = model(x_tensor).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()

        return [1 if prob > 0.5 else 0 for prob in probs]
    
    else:
        # Pair each single 1024-dim test embedding with all Aelfric train embeddings
        preds = []
        for test_emb in x_new:
            pred = predict_single_embedding(model, test_emb, a_train_embeddings)
            preds.append(pred)
        return preds
    
"""
Predicts the class (+1 for Aelfric, 0 for Unknown) for a single embedding by pairing it with a sample of Aelfric embeddings and averaging the model's scores.
(MIGHT CHANGE HOW WE AVERAGE SCORES INSTEAD OF JUST TAKING THE MEAN, MAYBE WEIGHT BY CONFIDENCE OR SOMETHING)

model: The trained LinearBinaryModel.
test_emb: A single embedding to classify (np.ndarray of shape (1024,)).
a_train_embeddings: A list of Aelfric training embeddings to pair with the test embedding for prediction.
Returns: A single predicted class (+1 for Aelfric, 0 for Unknown) for the test embedding based on the average score from the model.
"""
def predict_single_embedding(model: LinearBinaryModel, test_emb: np.ndarray, a_train_embeddings: list[np.ndarray]) -> int:
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
    return 1 if avg_score > config["threshold"] else 0


def log_test_data(model, x_train, y_train, x_test, y_test, a_train_embeddings, y_pred):
    global accuracy_data

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

    logging.debug("\n===== DATA DISTRIBUTION =====")

    logging.debug("Train:")
    logging.debug(f"\tAelfric: {np.sum(y_train == 1)}")
    logging.debug(f"\tUnknown: {np.sum(y_train == 0)}")
    logging.debug(f"\tTrain embeddings shape: {x_train.shape}")
    logging.debug(f"\tTrain embeddings:\n{x_train[:5][:5]}")
    logging.debug(f"\tTrain labels:\n{y_train[:5]}")

    logging.debug("Test:")
    logging.debug(f"\tAelfric: {np.sum(y_test == 1)}")
    logging.debug(f"\tUnknown: {np.sum(y_test == 0)}")

    logging.debug(f"\tAelfric training embeddings: {len(a_train_embeddings)}")
    logging.debug(f"\tAelfric test embeddings: {np.sum(y_test == 1)}")
    logging.debug(f"\tUnknown test embeddings: {np.sum(y_test == 0)}")
    logging.debug(f"\tAelfric training embeddings shape: {a_train_embeddings.shape}")
    #logging.debug(f"\tAelfric training embeddings:\n{a_train_embeddings[:5]}")

    logging.info("\n===== RAW SCORE DISTRIBUTION =====")
    
    logging.info("\nAelfric:")
    logging.info(f"\tMean: {np.mean(aelfric_scores):.4f}")
    logging.info(f"\tMin:  {np.min(aelfric_scores):.4f}")
    logging.info(f"\tMax:  {np.max(aelfric_scores):.4f}")
    logging.info(f"\tStd:  {np.std(aelfric_scores):.4f}")

    logging.info("\nUnknown:")
    logging.info(f"\tMean: {np.mean(unknown_scores):.4f}")
    logging.info(f"\tMin:  {np.min(unknown_scores):.4f}")
    logging.info(f"\tMax:  {np.max(unknown_scores):.4f}")
    logging.info(f"\tStd:  {np.std(unknown_scores):.4f}")

    logging.info("\n===== PREDICTIONS =====")
    
    #"""
    logging.debug("Individual scores:")

    logging.debug("\nAelfric:")
    for index, score in enumerate(aelfric_scores):
        if index < 5:  # Print only the first few scores
            logging.debug(f"  {score:.4f}")

    logging.debug("\nUnknown:")
    for index, score in enumerate(unknown_scores):
        if index < 5:  # Print only the first few scores
            logging.debug(f"  {score:.4f}")
    #"""

    logging.info("\nPrediction vs Actual:")

    accuracy_data["predicted_aelfric"] = np.sum(np.array(y_pred) == 1)
    accuracy_data["predicted_unknown"] = np.sum(np.array(y_pred) == 0)
    logging.info(f"Predicted Aelfric: {accuracy_data['predicted_aelfric']}")
    logging.info(f"Predicted Unknown: {accuracy_data['predicted_unknown']}")

    accuracy_data["actual_aelfric"] = np.sum(np.array(y_test) == 1)
    accuracy_data["actual_unknown"] = np.sum(np.array(y_test) == 0)
    logging.info(f"Actual Aelfric: {accuracy_data['actual_aelfric']}")
    logging.info(f"Actual Unknown: {accuracy_data['actual_unknown']}")

    logging.info("\nConfusion Matrix:") 

    accuracy_data["true_aelfric"] = np.sum((np.array(y_test) == 1) & (np.array(y_pred) == 1))
    accuracy_data["false_aelfric"] = np.sum((np.array(y_test) == 1) & (np.array(y_pred) == 0))
    logging.info(f"True Aelfric: {accuracy_data['true_aelfric']}")
    logging.info(f"False Aelfric: {accuracy_data['false_aelfric']}")

    accuracy_data["true_unknown"] = np.sum((np.array(y_test) == 0) & (np.array(y_pred) == 0))
    accuracy_data["false_unknown"] = np.sum((np.array(y_test) == 0) & (np.array(y_pred) == 1))
    logging.info(f"True Unknown: {accuracy_data['true_unknown']}")
    logging.info(f"False Unknown: {accuracy_data['false_unknown']}")


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


def save_to_spreadsheet():
    headers = [
        "date",
        "time",
        "author",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "input_dim",
        "hidden_dim",
        "dropout_rate",
        "num_epochs",
        "batch_size",
        "test_split",
        "random_seed",
        "num_samples",
        "threshold",
        "predicted_aelfric",
        "predicted_unknown",
        "actual_aelfric",
        "actual_unknown",
        "true_aelfric",
        "false_aelfric",
        "true_unknown",
        "false_unknown",
        "overall_accuracy",
        "aelfric_accuracy",
        "unknown_accuracy"
    ]
    row_data = [ datetime.now().strftime("%m/%d"), datetime.now().strftime("%H:%M") ]

    for header in headers:
        if header == "date" or header == "time":
            continue  # Already added date
        elif header in config:
            row_data.append( str(config[header]) if config[header] is not None else "N/A" )
        elif header in accuracy_data:
            row_data.append( str(accuracy_data[header]) if accuracy_data[header] is not None else "N/A" )
        else:
            row_data.append("N/A")

    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    credentials = Credentials.from_service_account_file(
        "credentials.json",
        scopes=SCOPES
    )

    client = gspread.authorize(credentials)
    spreadsheet = client.open("Training Model Results")
    worksheet = spreadsheet.sheet1

    row_index = len(worksheet.get_all_values()) + 1
    #worksheet.update(f"A{row_index}", [row_data])
    worksheet.update(values=[row_data], range_name=f"A{row_index}")


"""
Save the config and model to a file
"""
def save_model(model, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config
    }, path)
    logging.info(f"\nModel saved to {path}") 


"""
Load the config and model from a file
"""
def load_model(path):
    global config

    checkpoint = torch.load(path, map_location=device)

    current_threshold = config["threshold"]
    config = checkpoint["config"].copy()
    config["threshold"] = current_threshold

    model = LinearBinaryModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(f"Loaded model from {path}")

    return model


def generate_heatmap(model):
    x_train, x_test, y_train, y_test, _, _, _ = load_data()

    a_train_embeddings = x_train[np.array(y_train) == 1]  # Select train embeddings where label is 1 (Aelfric)
    a_test_embeddings = x_test[np.array(y_test) == 1]  # Select test embeddings where label is 1 (Aelfric)
    u_train_embeddings = x_train[np.array(y_train) == 0]  # Select train embeddings where label is 0 (Unknown)
    u_test_embeddings = x_test[np.array(y_test) == 0]  # Select test embeddings where label is 0 (Unknown)
    all_embeddings = np.concatenate((a_train_embeddings, a_test_embeddings, u_train_embeddings, u_test_embeddings))

    n = len(a_train_embeddings) + len(a_test_embeddings) + len(u_train_embeddings) + len(u_test_embeddings)
    heatmap = np.zeros((n, n))

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        for i, emb_i in tqdm(enumerate(all_embeddings)):
            for j, emb_j in enumerate(all_embeddings):
                pair = np.concatenate([emb_i, emb_j])
                logit = model(torch.tensor(pair, dtype=torch.float32).unsqueeze(0).to(device))
                pred = torch.sigmoid(logit).item()
                heatmap[i, j] = pred

    plt.figure(figsize=(12, 10))
    plt.imshow(heatmap, vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(label="Probability")
    plt.xlabel("Embedding")
    plt.ylabel("Embedding")
    plt.title("Pairwise Model Probabilities")
    logging.info("Saving heatmap...")
    plt.savefig("data/visualizations/heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()
    logging.info("Heatmap saved.")


def main():
    global accuracy_data, model_name, model_loaded

    if model_loaded:
        #model_path = "models/linear_model (94_75_99) (09_22_2026 11-00).pth"  # Update with your model path
        model = load_model(model_name)

        # Prepare the training and testing data
        x_train, y_train, x_test, y_test, a_train_embeddings = prepare_data()

        # Find and print accuracies
        y_pred = predict_linear_model(model, x_test, a_train_embeddings)

        acc_overall = accuracy(y_test, y_pred)
        class_acc = class_accuracy(y_test, y_pred)

        #log_test_data(model, x_train, y_train, x_test, y_test, a_train_embeddings, y_pred, acc_overall, class_acc)

        logging.info("\n===== Test Accuracies =====\n")
        logging.info(f"Overall test accuracy: {acc_overall*100:.2f}%")
        logging.info(f"Aelfric accuracy: {class_acc['Aelfric']*100:.2f}%")
        logging.info(f"Unknown accuracy: {class_acc['Unknown']*100:.2f}%")

        # Save the model with descriptive filename
        accuracy_data["overall_accuracy"] = f"{round(acc_overall*100, 2)}%"
        accuracy_data["aelfric_accuracy"] = f"{round(class_acc['Aelfric']*100, 2)}%"
        accuracy_data["unknown_accuracy"] = f"{round(class_acc['Unknown']*100, 2)}%"

        #save_to_spreadsheet()

    else:
        # Train the model
        x_train, y_train, x_test, y_test, a_train_embeddings = prepare_data()    

        model, in_accs, out_accs = train_linear_model(x_train, y_train, x_test, y_test, a_train_embeddings)
        logging.info("Training complete.")

        # Find and print accuracies
        y_pred = predict_linear_model(model, x_test, a_train_embeddings)

        acc_overall = accuracy(y_test, y_pred)
        class_acc = class_accuracy(y_test, y_pred)

        log_test_data(model, x_train, y_train, x_test, y_test, a_train_embeddings, y_pred)

        logging.info("\n===== Test Accuracies =====\n")
        logging.info(f"Overall test accuracy: {acc_overall*100:.2f}%")
        logging.info(f"Aelfric accuracy: {class_acc['Aelfric']*100:.2f}%")
        logging.info(f"Unknown accuracy: {class_acc['Unknown']*100:.2f}%")

        # Save the model with descriptive filename
        accuracy_data["overall_accuracy"] = f"{round(acc_overall*100, 2)}%"
        accuracy_data["aelfric_accuracy"] = f"{round(class_acc['Aelfric']*100, 2)}%"
        accuracy_data["unknown_accuracy"] = f"{round(class_acc['Unknown']*100, 2)}%"

        acc_str = f"{accuracy_data['overall_accuracy'][:-3]}_{accuracy_data['aelfric_accuracy'][:-3]}_{accuracy_data['unknown_accuracy'][:-3]}"
        timestamp_str = datetime.now().strftime("%m_%d_%Y %H-%M")
        model_name = f"models/linear_model ({acc_str}) ({timestamp_str}).pth" 

        save_model(model, model_name)
        save_to_spreadsheet()  # Save the model, config, and accuracy data to a spreadsheet

        model_loaded = True

        # Visualizations
        #in_errors = error(in_accs)
        #out_errors = error(out_accs)

        #visualization(model, in_errors, "In Sample Error", "inSample", "")
        #visualization(model, out_errors, "Out of Sample Error", "outSample", "")
        #visualization(model, {"In Sample Error": in_errors["Error"], "Out of Sample Error": out_errors["Error"]}, "Both Error", "bothSample", "")
        #generate_heatmap(model)


if __name__ == '__main__':
    for i in range(1):
        for j in range(3):
            main()
            logging.info(f"Completed iteration {j+1}/3")
            config["threshold"] += 0.1  # Increment the threshold for the next iteration

        config["threshold"] = 0.3
        config["test_split"] += 0.1
        model_loaded = False  # Reset model_loaded for each iteration