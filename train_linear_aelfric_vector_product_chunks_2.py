import os
import sys
import numpy as np
import json
from pathlib import Path
import logging
import torch
import torch.nn as nn
import random
import sklearn
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
device = torch.device("cpu")

class LinearBinaryModel(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, dropout_rate=0.2):
        super().__init__()
        self.use_hidden = bool(hidden_dim)  # False for 0 or None

        if self.use_hidden:
            self.hidden = nn.Linear(input_dim, hidden_dim)  # input → hidden
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout_rate)
            self.output = nn.Linear(hidden_dim, 1)          # hidden → output
        else:
            self.output = nn.Linear(input_dim, 1)           # input → output (logistic regression)

    def forward(self, x):
        if self.use_hidden:
            x = self.hidden(x)
            x = self.relu(x)
            x = self.dropout(x)
        return self.output(x)

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
LOAD_ORDER = "windows"  # "windows" = the order NTFS lists files, reproduced on any OS (default)
                        # "name"    = plain sort by file name
                        # "native"  = whatever this machine's filesystem returns (the old behavior)

def list_json_files(results_dir: str) -> list:
    files = list(Path(results_dir).glob('*.json'))
    if LOAD_ORDER == "native":
        return files
    if LOAD_ORDER == "name":
        return sorted(files, key=lambda p: p.name)
    # NTFS keeps directory entries sorted case-insensitively, by the uppercased name.
    # Sorting the same way gives a Windows machine's order on a Mac or Linux machine too.
    return sorted(files, key=lambda p: (p.name.upper(), p.name))

def load_embeddings(results_dir: str, max_num_of_texts: int = 500, details: bool = False) -> tuple:
    embeddings: list = []
    ids: list = []
    infos: list = []  # with details=True: file name and text of each embedding
    num_of_texts = 0

    # Go through every json file in the directory, in a fixed order
    for f in tqdm(list_json_files(results_dir)):
        with open(f, 'r', encoding='utf-8') as file:
            # Store the embeddings and ids associated with them
            data = json.load(file)

            job_id = next(iter(data))  # Get the job_id (Same as list(data.keys())[0])
            embeddings.append(data[job_id]['embedding']) # Appends text embedding
            text_id = job_id.split('_')[1] # Extracts text ID from job_id
            ids.append(text_id) # Appends text ID to the list of IDs
            if details:
                infos.append({"file": Path(f).name, "text": data[job_id].get('text')})

        num_of_texts += 1
        if num_of_texts > max_num_of_texts:
            break
                
    # Return the embeddings and ids (and details if asked for)
    if details:
        return np.array(embeddings), ids, infos
    return np.array(embeddings), ids

"""
Splits the TEXT IDs into training and testing texts. Only IDs are split, never embeddings or chunks,
so every chunk of a text lands on the same side as its text.

If config["split_file"] exists, the saved split is reused exactly (same texts, same order).
Otherwise the IDs are split with train_test_split (stratified by label) and, if export_split is on, saved.
Returns: id_train, id_test
"""
def split_ids(config: dict, ids: list, y: list) -> tuple[list, list]:
    if len(set(ids)) != len(ids):
        raise ValueError(f"Text IDs are not unique ({len(set(ids))} unique of {len(ids)}), so texts cannot be split by ID.")

    split_path = Path(config["split_file"]) if config.get("split_file") else None

    if split_path and split_path.exists():
        with open(split_path, 'r', encoding='utf-8') as f:
            split = json.load(f)

        # A split file silently overriding test_split / random_seed would be confusing, so refuse a mismatch
        for key in ("test_split", "random_seed"):
            if key in split and split[key] != config[key]:
                raise ValueError(f"{split_path} was made with {key}={split[key]} but the config has {key}={config[key]}. "
                                 "Use a different split_file name, or delete the file to create a new split.")

        known = set(ids)
        missing = [t for t in split["id_train"] + split["id_test"] if t not in known]
        if missing:
            raise ValueError(f"{len(missing)} IDs in {split_path} are not in the data, e.g. {missing[:5]}")
        return list(split["id_train"]), list(split["id_test"])

    # Split the IDs only. This gives the same split as splitting the embeddings did before,
    # because train_test_split shuffles by position and the IDs are in the same order.
    id_train, id_test, _, _ = train_test_split(ids, y, test_size=config["test_split"],
                                               random_state=config["random_seed"], stratify=y)

    # Write the split only when a split_file is named and export_split is on, and never over an existing file
    if split_path and config.get("export_split"):
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump({"test_split": config["test_split"], "random_seed": config["random_seed"],
                       "id_train": id_train, "id_test": id_test}, f)
    return list(id_train), list(id_test)


"""
Loads every chunk embedding in <aelfric_path>/chunks and <unknown_path>/chunks whose text ID is in wanted_ids,
grouped by text and sorted by chunk index.
Returns: {text_id: [(chunk embedding, details), ...]}, details = {text_id, chunk_index, file, text}
"""
def load_chunks_by_text(config: dict, wanted_ids: set) -> dict:
    grouped: dict = {}
    for folder in (config["aelfric_path"], config["unknown_path"]):
        embs, parents, indexes, infos = load_chunk_embeddings(str(Path(folder) / "chunks"), details=True)
        for emb, parent, idx, info in zip(embs, parents, indexes, infos):
            if parent in wanted_ids:
                info = {"text_id": parent, "chunk_index": idx, "file": info["file"], "text": info["text"]}
                grouped.setdefault(parent, []).append((idx if idx is not None else 0, emb, info))

    missing = [t for t in wanted_ids if t not in grouped]
    if missing:
        print(f"WARNING: {len(missing)} texts have no chunk files and are left out, e.g. {sorted(missing)[:5]}")

    return {t: [(emb, info) for _, emb, info in sorted(chunks, key=lambda c: c[0])] for t, chunks in grouped.items()}


"""
Collects the chunks of the given texts, in the order of text_ids and then chunk index.
Each chunk takes its text's label.
Returns: chunk embeddings (np.ndarray), chunk labels, the text ID of each chunk, and each chunk's details.
"""
def gather_chunks(text_ids: list, chunks_by_text: dict, label_by_id: dict) -> tuple[np.ndarray, list, list, list]:
    embs, labels, parents, infos = [], [], [], []
    for t in text_ids:
        for emb, info in chunks_by_text.get(t, []):
            embs.append(emb)
            labels.append(label_by_id[t])
            parents.append(t)
            infos.append(info)
    return np.array(embs), labels, parents, infos


"""
Loads the data and splits it by TEXT ID into training and testing sets.

Pulled from the config:
  aelfric_path, unknown_path: folders of the whole-text embeddings (chunks are in their "chunks" subfolders)
  train_on_chunks: True = return chunk embeddings, False = return whole-text embeddings (as before)
  test_split, random_seed, split_file, export_split: how the IDs are split (see split_ids)

The text IDs always come from the whole-text files, so the split is the same in both modes and split_ids.json is shared.

Returns:
  x_train: embeddings of the training items (chunks of the training texts, or the training text embeddings)
  x_test:  embeddings of the testing items (chunks of the testing texts, or the testing text embeddings)
  y_train: label of each training item (+1 Aelfric, 0 Unknown); a chunk takes its text's label
  y_test:  label of each testing item
  id_train: IDs of the full texts in the training split (Aelfric and Unknown)
  id_test:  IDs of the full texts in the testing split (Aelfric and Unknown)
  parents_train: text ID of each item in x_train (needed to keep Aelfric-Aelfric pairs from different texts)
  parents_test:  text ID of each item in x_test (needed to combine chunk scores into one result per text)
  info_train, info_test: details of each item, row for row:
      {"text_id", "chunk_index", "file", "text"} (chunk_index is None for a whole text)
In text mode each item is a whole text, so parents_train == id_train and parents_test == id_test.
"""
def load_data(config: dict) -> tuple:
    # Whole-text files: these give the text IDs and labels (and the embeddings in text mode)
    a_embeddings, a_ids, a_infos = load_embeddings(config["aelfric_path"], config["num_samples"], details=True)
    u_embeddings, u_ids, u_infos = load_embeddings(config["unknown_path"], config["num_samples"], details=True)

    ids = a_ids + u_ids
    y = [1] * len(a_ids) + [0] * len(u_ids)
    label_by_id = dict(zip(ids, y))

    # Split the IDs only
    id_train, id_test = split_ids(config, ids, y)

    if config.get("train_on_chunks"):
        chunks_by_text = load_chunks_by_text(config, set(id_train) | set(id_test))
        x_train, y_train, parents_train, info_train = gather_chunks(id_train, chunks_by_text, label_by_id)
        x_test, y_test, parents_test, info_test = gather_chunks(id_test, chunks_by_text, label_by_id)
    else:
        emb_by_id = dict(zip(ids, np.concatenate((a_embeddings, u_embeddings), axis=0)))
        x_train = np.array([emb_by_id[t] for t in id_train])
        x_test = np.array([emb_by_id[t] for t in id_test])
        y_train = [label_by_id[t] for t in id_train]
        y_test = [label_by_id[t] for t in id_test]
        parents_train, parents_test = list(id_train), list(id_test)
        info_by_id = {t: {"text_id": t, "chunk_index": None, **i} for t, i in zip(ids, a_infos + u_infos)}
        info_train = [info_by_id[t] for t in id_train]
        info_test = [info_by_id[t] for t in id_test]

    return x_train, x_test, y_train, y_test, id_train, id_test, parents_train, parents_test, info_train, info_test


"""
Create Aelfric-Aelfric and Aelfric-Unknown pairs for regular embedding training not chunk training. 

embs: List of embeddings (np.ndarray).
labels: List of corresponding labels (1 for Aelfric, 0 for Unknown).
num_samples: The number of samples to draw from each class.
Returns: Two lists of concatenated embeddings.
"""
def pair_embeddings(embs: list[np.ndarray], labels: list[int], num_samples:int = 250, random_seed: int = 42) -> tuple[list[np.ndarray], list[np.ndarray]]:
    a_embs = []
    u_embs = []

    for i, emb in enumerate(embs):
        if labels[i] == 1:
            a_embs.append(emb)
        else:
            u_embs.append(emb)

    random.seed(random_seed)  # Ensure reproducibility for sampling
    a_sample = random.sample(a_embs, k=min(num_samples, len(a_embs))) ### Eventually the samples need to be controlled better
    random.seed(random_seed)  # Ensure reproducibility for sampling
    u_sample = random.sample(u_embs, k=min(num_samples, len(u_embs)))

    aa_pairs = []
    au_pairs = []

    # Aelfric–Aelfric pairs (1)
    for emb_a in a_sample:
        for emb_b in a_sample:
            aa_pairs.append(np.multiply(emb_a, emb_b))

    # Aelfric–Unknown pairs (0)
    for emb_a in a_sample:
        for emb_b in u_sample:
            au_pairs.append(np.multiply(emb_a, emb_b))

    # Unknown-Aelfric pairs (0)
    for emb_a in u_sample:
        for emb_b in a_sample:
            au_pairs.append(np.multiply(emb_a, emb_b))

    random.seed(random_seed)  # Ensure reproducibility for sampling
    au_sample = random.sample(au_pairs, k=min(num_samples*3, len(au_pairs)))  # Sample to limit size

    return aa_pairs, au_sample


"""
Builds training pairs from chunks so it should make around 12,000 A-A pairs (label 1) 
and 12,000 A-U pairs (label 0).

Aelfric-Aelfric pairs always come from two different texts.

The two embedding vectors are then multiplied into one vector with np.multiply due to this model's method being 
vector products.
"""
def pair_chunk_embeddings(embs: np.ndarray, labels: list, parents: list, n_aa: int, n_au: int, random_seed: int = 42) -> tuple[list, list]:
    a_idx = [i for i, l in enumerate(labels) if l == 1]
    u_idx = [i for i, l in enumerate(labels) if l == 0]
    rng = random.Random(random_seed)

    def sample_pairs(first, second, n, same_class):
        pairs, seen, attempts = [], set(), 0
        while len(pairs) < n and attempts < n * 50:
            attempts += 1
            i, j = rng.choice(first), rng.choice(second)
            if same_class:
                if parents[i] == parents[j]:
                    continue
                key = (min(i, j), max(i, j))
            else:
                key = (i, j)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(np.multiply(embs[i], embs[j]))
        return pairs

    aa_pairs = sample_pairs(a_idx, a_idx, n_aa, True)
    au_pairs = sample_pairs(a_idx, u_idx, n_au, False)
    return aa_pairs, au_pairs


"""
Loads the data and prepares training pairs

After calling pair_chunk_embeddings it gives the Aelfric-Aelfric and the Aelfric-Unknown pairs.
Those then become the pairs that are put into x_train and y_train

"""
def prepare_data(config: dict) -> tuple:
    x_train_old, x_test, y_train_old, y_test, id_train, id_test, parents_train, parents_test, _, info_test = load_data(config)

    # Aelfric reference pool for scoring: Aelfric training chunks in chunk mode, Aelfric training texts otherwise
    a_train_embeddings = x_train_old[np.array(y_train_old) == 1]

    unit = "chunks" if config.get("train_on_chunks") else "texts"
    print("Loaded embeddings...")
    print(f"Train {unit}: {len(y_train_old)} ({sum(y_train_old)} Aelfric, {len(y_train_old) - sum(y_train_old)} Unknown) from {len(id_train)} texts")
    print(f"Test {unit}: {len(y_test)} ({sum(y_test)} Aelfric, {len(y_test) - sum(y_test)} Unknown) from {len(id_test)} texts")

    if config.get("train_on_chunks"):
        aa_pairs, au_pairs = pair_chunk_embeddings(x_train_old, y_train_old, parents_train,
                                                   n_aa=config.get("chunk_aa_pairs", 12000), n_au=config.get("chunk_au_pairs", 12000),
                                                   random_seed=config["random_seed"])
        print(f"Chunk pairs: {len(aa_pairs)} Aelfric-Aelfric, {len(au_pairs)} Aelfric-Unknown")
    else:
        aa_pairs, au_pairs = pair_embeddings(x_train_old, y_train_old, num_samples=config["num_samples"], random_seed=config["random_seed"])

    # --- Combine and label training pairs ---
    x_aa = np.stack(aa_pairs)
    x_au = np.stack(au_pairs)

    y_aa = np.ones(len(x_aa))   # +1 for Aelfric–Aelfric
    y_au = np.zeros(len(x_au))  # 0 for Aelfric–Unknown

    x_train = np.vstack([x_aa, x_au])
    y_train = np.concatenate([y_aa, y_au])

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), a_train_embeddings, id_test, parents_test, info_test


"""
Creates the model, trains it on the training data, and evaluates it on both training and testing data.
The chunk pairs of A-A and A-U are used to train weights but aren't used for the actual testing.

x_train: The chunk pair vectors so A-A and A-U combined into one
y_train: Training labels of the paired vectors
x_test: The test chunks (whole text embeddings when train_on_chunks is False)
y_test: The test chunk labels, each chunk taking its text's label (whole text labels when train_on_chunks is False)
a_train_embeddings: using the Aelfric training chunks to score test items

Returns: The trained model along with dictionaries containing in-sample and out-of-sample accuracies for overall, Aelfric, and Unknown classes.
"""
def train_linear_model(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, a_train_embeddings: np.ndarray, config: dict) -> tuple[LinearBinaryModel, dict, dict]:
    random.seed(config["random_seed"]); np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    x_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)  # shape (N_total, 1)

    #creating the model 
    model = LinearBinaryModel(input_dim=config["input_dim"], hidden_dim=config["hidden_dim"], dropout_rate=config["dropout_rate"]).to(device)

    #This is added to penalize mistakes on Aelfric labels by weighing them heavier
    pos_weight = torch.tensor([np.sum(y_train == 0) / np.sum(y_train == 1)], dtype=torch.float32).to(device)  # Unknown/Aelfric ratio
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # AdamW optimizer with weight decay and learning
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

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
        in_predictions = predict_linear_model(model, x_train, config=config)
        in_acc_total.append(accuracy(y_train, in_predictions)*100)
        in_accs = class_accuracy(y_train, in_predictions)
        in_acc_aelfric.append(in_accs["Aelfric"]*100)
        in_acc_unknown.append(in_accs["Unknown"]*100)

        out_predictions = predict_linear_model(model, x_test, a_train_embeddings, config=config)
        out_acc_total.append(accuracy(y_test, out_predictions)*100)
        out_accs = class_accuracy(y_test, out_predictions)
        out_acc_aelfric.append(out_accs["Aelfric"]*100)
        out_acc_unknown.append(out_accs["Unknown"]*100)

    return model, \
           {"Accuracy": in_acc_total, "Aelfric Accuracy": in_acc_aelfric, "Unknown Accuracy": in_acc_unknown}, \
           {"Accuracy": out_acc_total, "Aelfric Accuracy": out_acc_aelfric, "Unknown Accuracy": out_acc_unknown}


"""
Predicts the class (+1 for Aelfric, 0 for Unknown) for new embeddings using the trained model.
If Aelfric embeddings are provided, it pairs each new embedding with all Aelfric embeddings to make predictions based on the average score
which is all of the arrays from multiplying with the 100 Aelfric refrences and averaging them.

Only draws from training chunks for the Aelfric references.

These then go to the neural network.

model: The trained LinearBinaryModel.
x_new: New embeddings to classify 
a_train_embeddings: Optional Aelfric embeddings to pair with new embeddings for prediction.
Returns: A list of predicted classes (+1 for Aelfric, 0 for Unknown) for each new embedding.
"""
def predict_linear_model(model: LinearBinaryModel, x_new: np.ndarray, a_train_embeddings: np.ndarray = None, config: dict = None) -> list[int]:
    if a_train_embeddings is None:
        if x_new.ndim == 1:
            x_new = x_new[np.newaxis, :]

        model.eval()

        with torch.no_grad():
            x_tensor = torch.tensor(x_new, dtype=torch.float32).to(device)
            logits = model(x_tensor).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()

        #Aelfric Threshold Value where it can be changed to either accept more or less values as Aelfric
        threshold = config["threshold"] if config else 0.4
        return [1 if prob > threshold else 0 for prob in probs]

    else:
        scores = predict_scores(model, x_new, a_train_embeddings, random_seed=config["random_seed"])
        return [1 if s > config["threshold"] else 0 for s in scores]


"""
Picks the Aelfric reference sample: the same 100 as predict_single_chunk picks for the same seed
(random.sample chooses positions from the length alone, so sampling positions gives the same picks).

The same 100 reference chunks are used for all testing 
"""
def reference_sample(a_train_embeddings: np.ndarray, random_seed: int, k: int = 100) -> np.ndarray:
    random.seed(random_seed)
    positions = random.sample(range(len(a_train_embeddings)), k=min(len(a_train_embeddings), k))
    return np.asarray(a_train_embeddings)[positions]


"""
Average Aelfric probability of every item in x_new, against the same 100 Aelfric references.
Gives the same scores as calling predict_single_chunk on each item, but picks the references once
and scores many items per pass, which matters with thousands of test chunks.
Returns: np.ndarray of one score per item.
"""
def predict_scores(model: LinearBinaryModel, x_new: np.ndarray, a_train_embeddings: np.ndarray, random_seed: int = 42, batch_size: int = 64) -> np.ndarray:
    refs = reference_sample(a_train_embeddings, random_seed)
    x_new = np.asarray(x_new)
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(x_new), batch_size):
            batch = x_new[start:start + batch_size]
            pairs = (batch[:, None, :] * refs[None, :, :]).reshape(-1, batch.shape[1])  # each item x each reference
            logits = model(torch.tensor(pairs, dtype=torch.float32).to(device))
            probs = torch.sigmoid(logits).cpu().numpy().reshape(len(batch), len(refs))
            scores.extend(probs.mean(axis=1).tolist())
    return np.array(scores)


"""
Combines all the chunks scores for a text into one decision per text: the mean of a text's chunk scores, compared with the threshold.

Returns: text IDs, their labels, and their predictions.
"""
def text_level_predictions(scores: np.ndarray, labels: list, parents: list, threshold: float) -> tuple[list, list, list]:
    by_text: dict = {}
    label_by_text: dict = {}
    for s, l, t in zip(scores, labels, parents):
        by_text.setdefault(t, []).append(s)
        label_by_text[t] = l
    texts = list(by_text)
    return texts, [label_by_text[t] for t in texts], [1 if np.mean(by_text[t]) > threshold else 0 for t in texts]


"""
Logs overall, Aelfric, and Unknown accuracy under a heading. Returns (overall accuracy, class accuracies).
"""
def report_accuracy(logger, heading: str, y_true: list, y_pred: list) -> tuple[float, dict]:
    acc, cls = accuracy(y_true, y_pred), class_accuracy(y_true, y_pred)
    logger.info(f"{heading}: overall {acc*100:.2f}% | Aelfric {cls['Aelfric']*100:.2f}% | Unknown {cls['Unknown']*100:.2f}% ({len(y_true)} items)")
    return acc, cls


"""
Scores the test items and reports the results. In chunk mode: chunk level, then text level (mean of chunk scores).
In text mode: text level only.
If csv_path is given, every test item's details, score and prediction are written there (see save_predictions).
Returns: (overall, class accuracies) at text level, used for the checkpoint name.
"""
def evaluate(model, config: dict, x_test, y_test, parents_test, a_train_embeddings, logger, chunks: bool,
             info_test: list = None, csv_path: str = None) -> tuple[float, dict]:
    scores = predict_scores(model, x_test, a_train_embeddings, random_seed=config["random_seed"])
    threshold = config["threshold"]
    if chunks:
        report_accuracy(logger, f"Chunk level (threshold {threshold})", list(y_test), [1 if s > threshold else 0 for s in scores])
    _, t_labels, t_pred = text_level_predictions(scores, list(y_test), list(parents_test), threshold)
    heading = "Text level, mean of chunk scores" if chunks else "Text level"
    result = report_accuracy(logger, f"{heading} (threshold {threshold})", t_labels, t_pred)
    if csv_path and info_test is not None:
        save_predictions(csv_path, info_test, list(y_test), scores, list(parents_test), threshold)
        logger.info(f"Predictions saved to {csv_path}")
    return result


"""
Writes one row per test item (chunk or text) to a CSV file: text ID, chunk index, file name, true label,
score, prediction, whether it was correct, the text's mean score and text-level prediction, and the passage itself.
Open it in Excel or Sheets and filter on correct = no to read the misclassified passages.
"""
def save_predictions(path: str, info_test: list, y_test: list, scores, parents_test: list, threshold: float) -> None:
    import csv
    by_text: dict = {}
    for s, t in zip(scores, parents_test):
        by_text.setdefault(t, []).append(s)
    text_score = {t: float(np.mean(v)) for t, v in by_text.items()}
    name = lambda l: "Aelfric" if l == 1 else "Unknown"

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:   # utf-8-sig so Excel shows Old English letters
        w = csv.writer(f)
        w.writerow(["text_id", "chunk_index", "file", "label", "score", "prediction", "correct",
                    "text_score", "text_prediction", "text_correct", "passage"])
        for info, label, s in zip(info_test, y_test, scores):
            pred = 1 if s > threshold else 0
            ts = text_score[info["text_id"]]
            tpred = 1 if ts > threshold else 0
            w.writerow([info["text_id"], "" if info["chunk_index"] is None else info["chunk_index"], info["file"],
                        name(label), f"{s:.4f}", name(pred), "yes" if pred == label else "no",
                        f"{ts:.4f}", name(tpred), "yes" if tpred == label else "no", info.get("text") or ""])


"""
Predicts the class for a single chunk by pairing it with a sample of Aelfric chunks and averaging the model's scores.
Not called anywhere now: predict_linear_model uses predict_scores, which gives the same results for many chunks at once.
Kept to show the steps for one chunk.

test_emb: The chunk that is being tested that is a vector of 1024 for embedding, this is not using A-A or A-U chunk pairs
a_train_embeddings: A list of Aelfric training embeddings or chunks to pair with for prediction scoring. 
Returns: A single predicted class (+1 for Aelfric, 0 for Unknown) for the test embedding based on the average. 

In pairs 100 known aelfric chunks are multiplied with the single test chunk that is passed to it outputing 100 vectors of 1024.
All the pairs are then combined into a 100 x 1024 array then are sent to tensor 
They are then all run through the network, converted into an array, then finally the avg creates the result 

"""
def predict_single_chunk(model: LinearBinaryModel, test_emb: np.ndarray, a_train_embeddings: list[np.ndarray], random_seed: int = 42, threshold = 0.5) -> int:
    
    # Sample fewer Aelfric embeddings for speed
    random.seed(random_seed)  # Ensure reproducibility for sampling
    aelfrics = random.sample(a_train_embeddings.tolist(), k=min(len(a_train_embeddings), 100))

    # Making 100 pair vectors which is the chunk being tested and the refrence embedding for the sample of 100 Aelfrics 
    pairs = [np.multiply(a, test_emb) for a in aelfrics]

    #stacks the 100 vectors into a 100 x 1024 array
    x_tensor = torch.tensor(np.stack(pairs), dtype=torch.float32).to(device)

    # Switches to evaluation mode
    model.eval()

    #this then scores all 100 pairs at once 
    with torch.no_grad():
        logits = model(x_tensor).squeeze()  # one score per pair
        probs = torch.sigmoid(logits).cpu().numpy()  # convert to probabilities

    # Then takes the average, sees if its above the threshold then decides 
    avg_score = np.mean(probs)  # average probability across all pairs
    return 1 if avg_score > threshold else 0


"""
Loads chunk-level embeddings from the "chunks" subfolder of a text-embedding folder
(for example data/.../Aelfric/chunks/B1.1.1_chunk_0.json).

Each chunk file holds the same job key as its parent text's embedding file, so the parent text ID is taken
the same way as in load_embeddings (job_id.split('_')[1]). That keeps chunk IDs identical to the text IDs in split_ids.json.
Returns: chunk embeddings (np.ndarray), the parent text ID of each chunk, and each chunk's index within its text.
"""
def load_chunk_embeddings(chunks_dir: str, details: bool = False) -> tuple:
    if not Path(chunks_dir).is_dir():
        raise FileNotFoundError(f"No chunk folder at {chunks_dir}")

    embeddings, parent_ids, chunk_indexes, infos = [], [], [], []
    for f in tqdm(list_json_files(chunks_dir)):
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)
        job_id = next(iter(data))
        embeddings.append(data[job_id]['embedding'])
        parent_ids.append(job_id.split('_')[1])
        chunk_indexes.append(data[job_id].get('chunk_index'))
        if details:
            infos.append({"file": Path(f).name, "text": data[job_id].get('chunk_text')})
    if details:
        return np.array(embeddings), parent_ids, chunk_indexes, infos
    return np.array(embeddings), parent_ids, chunk_indexes


def log_test_data(model, x_train, y_train, x_test, y_test, a_train_embeddings, y_pred, random_seed=42):
    model.eval()

    aelfric_scores = []
    unknown_scores = []

    scores = predict_scores(model, x_test, a_train_embeddings, random_seed=random_seed)
    y_true = np.array(y_test)
    aelfric_scores = list(scores[y_true == 1])
    unknown_scores = list(scores[y_true == 0])

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
This calculates the best threshold  
"""
def sweep_thresholds(model, x_test, y_test, a_train_embeddings, random_seed, thresholds=None):
    from sklearn.metrics import roc_auc_score
    thresholds = thresholds if thresholds is not None else np.arange(0.2, 0.81, 0.05)

    scores = predict_scores(model, x_test, a_train_embeddings, random_seed=random_seed)
    y = np.array(y_test)

    print(f"ROC AUC: {roc_auc_score(y, scores):.4f}")
    for th in thresholds:
        pred = (scores > th).astype(int)
        print(f"thr {th:.2f}: overall {np.mean(pred == y)*100:.2f}  "
              f"Aelfric {np.mean(pred[y == 1] == 1)*100:.2f}  Unknown {np.mean(pred[y == 0] == 0)*100:.2f}")
    return scores

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
        "config": config,
        # Library versions, so differences between machines can be traced
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "numpy": np.__version__, "scikit-learn": sklearn.__version__}
    }, path)
    print(f"Model saved to {path}") 


"""
Load the config and model from a file
"""
def load_model(path):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = LinearBinaryModel(input_dim=config["input_dim"], hidden_dim=config["hidden_dim"], dropout_rate=config["dropout_rate"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {path}")
    return config, model
    

def main():
    logger = setup_logging()

    # Git does not track empty folders, so a fresh clone (for example on Windows) may not have these
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/visualizations", exist_ok=True)
    logger.info(f"python {sys.version.split()[0]}, torch {torch.__version__}, "
                f"numpy {np.__version__}, scikit-learn {sklearn.__version__}")

    model_loaded = False  # Set to True if you want to load a pre-trained model instead of training a new one
    # For a text-trained model: also score the chunks of the test texts (needs a "chunks" folder inside each data folder).
    # A chunk-trained model is always scored on chunks.
    evaluate_on_chunks = False

    if model_loaded:
        # Must be a checkpoint trained with this file's np.multiply pairs (input_dim 1024).
        # The old concatenation checkpoints (input_dim 2048) won't work with this model
        model_path = "models/<your product checkpoint>.pth"
        config, model = load_model(model_path)
        config.setdefault("threshold", 0.5)    # older checkpoints have no threshold; use the one from your sheet
        config.setdefault("split_file", None)  # older checkpoints have no split file; set one to use the original test set

        # Only the test items and the Aelfric reference pool are needed, so skip building training pairs.
        # load_data returns chunks if the checkpoint's config says train_on_chunks, so the references match training.
        x_train, x_test, y_train, y_test, id_train, id_test, parents_train, parents_test, _, info_test = load_data(config)
        a_train_embeddings = x_train[np.array(y_train) == 1]

        pred_name = Path(model_path).stem
        evaluate(model, config, x_test, y_test, parents_test, a_train_embeddings, logger, chunks=bool(config.get("train_on_chunks")),
                 info_test=info_test, csv_path=f"data/predictions/{pred_name}.csv")

        if evaluate_on_chunks and not config.get("train_on_chunks"):
            # Text-trained model on chunks: chunks of the same test texts, scored against the Aelfric training TEXTS
            _, cx_test, _, cy_test, _, _, _, cparents_test, _, cinfo_test = load_data(dict(config, train_on_chunks=True, export_split=False))
            evaluate(model, config, cx_test, cy_test, cparents_test, a_train_embeddings, logger, chunks=True,
                     info_test=cinfo_test, csv_path=f"data/predictions/{pred_name} (chunks).csv")


    else:
        config = {
           "learning_rate": 3e-4,
           "weight_decay": 0.35,
           "input_dim": 1024,
           "hidden_dim": 50,
           "dropout_rate": 0.3,
           "num_epochs": 75,
           "batch_size": 32,
           "test_split": 0.3,
           "random_seed": 11,
           "num_samples": 1000,
           "threshold": 0.45,
           "aelfric_path": "data/sorted_embeddings/Train/Large Chunk/Aelfric",
           "unknown_path": "data/sorted_embeddings/Train/Large Chunk/Unknown",
           # The first run creates split_file; later runs reuse it, so every machine gets the same test set.
           # If you change test_split or random_seed, point split_file at a new name (or delete the old file).
           "split_file": "split_ids.json",
           "export_split": True,
           # True = train and test on the chunk embeddings in <path>/chunks; False = whole-text embeddings (as before)
           "train_on_chunks": True,
           "chunk_aa_pairs": 12000,   # Aelfric-Aelfric chunk pairs (always from two different texts)
           "chunk_au_pairs": 12000,   # Aelfric-Unknown chunk pairs
       }

        chunks = bool(config.get("train_on_chunks"))

        # Train the model
        x_train, y_train, x_test, y_test, a_train_embeddings, id_test, parents_test, info_test = prepare_data(config)

        # Now that we have the A-A and A-U chunk embedings we put them into the linear model
        # In chunk mode x_test holds the test CHUNKS, so the per-epoch out-of-sample accuracy is chunk level
        model, in_accs, out_accs = train_linear_model(x_train, y_train, x_test, y_test, a_train_embeddings, config)
        logger.info("Training complete.")

        y_pred = predict_linear_model(model, x_test, a_train_embeddings, config=config)
        log_test_data(model, x_train, y_train, x_test, y_test, a_train_embeddings, y_pred, random_seed=config["random_seed"])

        # Results: chunk level and text level in chunk mode, text level in text mode
        acc_overall, class_acc = evaluate(model, config, x_test, y_test, parents_test, a_train_embeddings, logger, chunks=chunks)

        # Save the model, named after its TEXT-level accuracy (comparable across chunk and text models)
        acc_str = f"{round(acc_overall*100)}_{round(class_acc['Aelfric']*100)}_{round(class_acc['Unknown']*100)}"
        timestamp_str = datetime.now().strftime("%m_%d_%Y %H-%M")
        model_name = "linear_model_chunks" if chunks else "linear_model"
        model_filename = f"models/{model_name} ({acc_str}) ({timestamp_str}).pth" 
        save_model(config, model, model_filename)

        # Every test item's score and prediction, with its passage, named after the checkpoint
        pred_name = Path(model_filename).stem
        save_predictions(f"data/predictions/{pred_name}.csv", info_test, list(y_test),
                         predict_scores(model, x_test, a_train_embeddings, random_seed=config["random_seed"]),
                         list(parents_test), config["threshold"])
        logger.info(f"Predictions saved to data/predictions/{pred_name}.csv")

        if evaluate_on_chunks and not chunks:
            _, cx_test, _, cy_test, _, _, _, cparents_test, _, cinfo_test = load_data(dict(config, train_on_chunks=True, export_split=False))
            evaluate(model, config, cx_test, cy_test, cparents_test, a_train_embeddings, logger, chunks=True,
                     info_test=cinfo_test, csv_path=f"data/predictions/{pred_name} (chunks).csv")

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