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

"""
Things to do:
- Improve line visualization function appearance?
- Fully implement chunk heatmap visualization classification accuracy
- Save the model and reload it for testing
- Try for better accuracy with different parameters
- 
"""

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class LinearBinaryModel(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=256):#, dropout_rate=0.05): 
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)  # input → hidden
        self.relu = nn.ReLU()                            # non-linearity
        #self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(hidden_dim, 1)          # hidden → output

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.output(x)   # scalar score
        return x

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
    

def visualization(model, lines : dict, title, filename, otherData):
    modelString = str(model)

    # Count lines to estimate height needed 
    modelLines = len(modelString.splitlines())
    extraHeight = 0.1 + modelLines * 0.015  # adjust for text length
    
    # Create figure with dynamic bottom margin
    fig, ax = plt.subplots(figsize=(15, 5 + extraHeight * 10))
    fig.subplots_adjust(bottom=(extraHeight * 10) / ((extraHeight * 10) + 5))

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
        0.1, 0.02, modelString,
        fontsize=10,
        va='bottom',
        ha='left',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    )
    # Add model description below the plot
    fig.text(
        0.5, 0.02, otherData,
        fontsize=10,
        va='bottom',
        ha='left',
        family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8)
    )

    plt.savefig(f"data/visualizations/{filename}.png")  


'''
# Return list of embeddings and ids
def load_embeddings(results_dir: str) -> dict:
    embeddings_dict = {}

    # Iterates through each embedding in the directory
    for f in Path(results_dir).glob('*.json'):
        with open(f, 'r') as file:
            data = json.load(file)
            for job_id, content in data.items():
                embeddings_dict[job_id] = np.array(content['embedding'], dtype=np.float32) 

    return embeddings_dict


def pair_embeddings(a_sample, u_sample):
    """
    Create Aelfric-Aelfric (positive) and Aelfric-Unknown (negative) pairs.
    Returns two lists of concatenated embeddings.
    """
    aelfric_aelfric_pairs = []
    aelfric_unknown_pairs = []

    # Aelfric–Aelfric pairs (positive)
    for emb_a in a_sample:
        for emb_b in a_sample:
            aelfric_aelfric_pairs.append(np.concatenate([emb_a, emb_b]))

    # Aelfric–Unknown pairs (negative)
    for emb_a in a_sample:
        for emb_b in u_sample:
            aelfric_unknown_pairs.append(np.concatenate([emb_a, emb_b]))

    return aelfric_aelfric_pairs, aelfric_unknown_pairs


def prepare_data():
    # --- Load individual embeddings ---
    aelfric_embeddings = load_embeddings("data/sorted_embeddings/Aelfric")
    unknown_embeddings = load_embeddings("data/sorted_embeddings/Unknown")

    # Convert to lists
    aelfric_list = list(aelfric_embeddings.values())
    unknown_list = list(unknown_embeddings.values())

    # --- Split each into train/test ---
    aelfric_train, aelfric_test = train_test_split(aelfric_list, test_size=0.2, random_state=0)
    unknown_train, unknown_test = train_test_split(unknown_list, test_size=0.2, random_state=0)

    # --- Create training pairs ---
    # (optionally sample to control training size)
    aelfric_train_sample = random.sample(aelfric_train, k=min(500, len(aelfric_train)))
    unknown_train_sample = random.sample(unknown_train, k=min(500, len(unknown_train)))

    aelfric_aelfric_pairs, aelfric_unknown_pairs = pair_embeddings(aelfric_train_sample, unknown_train_sample)

    # --- Combine and label training pairs ---
    X_a = np.stack(aelfric_aelfric_pairs)
    X_u = np.stack(aelfric_unknown_pairs)

    y_a = np.ones(len(X_a))   # +1 for Aelfric–Aelfric
    y_u = -np.ones(len(X_u))  # -1 for Aelfric–Unknown

    X_train = np.vstack([X_a, X_u])
    y_train = np.concatenate([y_a, y_u])

    # --- Prepare test data (not paired yet) ---
    X_test = np.array(aelfric_test + unknown_test)
    y_test = np.concatenate([np.ones(len(aelfric_test)), -np.ones(len(unknown_test))])

    return X_train, y_train, X_test, y_test, aelfric_train_sample
'''

# Return list of embeddings and ids
def load_embeddings(results_dir: str, allIDs = []):
    embeddings = []
    ids = []
    num_of_chunks = 0
    
    # Go through every json file
    for f in tqdm(Path(results_dir).glob('*.json')):
        with open(f, 'r') as file:
            # Store the embeddings and ids associated with them
            data = json.load(file)
            for job_id, content in data.items():
                embeddings.append(content['embedding'])
                
                textID = job_id.split('_')[1]
                chunkNumber = str(f).split('\\')[-1].split('_')[-1].split('.')[0]
                ids.append(f"{textID}_{chunkNumber}")

                # This will get the list of the overall texts
                if textID not in allIDs:
                    allIDs.append(textID)

        num_of_chunks += 1
        if num_of_chunks > 250:
            break
                
    # Return the embeddings and ids
    return np.array(embeddings), ids


def loadData(APath, UPath, trainSplit, randomSeed):
    # Will store the IDs of individual texts to keep chunks together
    allId = []
    allY = []
    # Get the Aelfric embeddings
    AEmbeddings, AIds = load_embeddings(APath, allId)
    allY = ([1] * len(allId))
    aelfricsize = len(allId)
    # Get the Unknown embeddings
    UEmbeddings, UIds = load_embeddings(UPath, allId)
    allY += ([-1] * (len(allId) - aelfricsize))

    # Create the list of all the embeddings and ids
    embeddings = np.concatenate((AEmbeddings, UEmbeddings), axis=0)
    y = [1] * len(AEmbeddings) + [-1] * len(UEmbeddings)
    ids =  AIds + UIds

    # Randomly split the full works
    trainingID, testingID, yid1, yid2 = train_test_split(allId, allY, test_size=trainSplit, random_state=randomSeed, stratify=allY)

    # Initialize all the lists
    xTrain = []
    xTest = []
    yTrain = []
    yTest = []
    idTrain = []
    idTest = []

    # Go through all of the chunks and add them to either the training or testing lists
    for i, embedding in enumerate(embeddings):
        if ids[i].split("_")[0] in trainingID:
            xTrain.append(embedding)
            yTrain.append(y[i])
            idTrain.append(ids[i])
        else:
            xTest.append(embedding)
            yTest.append(y[i])
            idTest.append(ids[i])

    return xTrain, xTest, yTrain, yTest, idTrain, idTest, y
    

def pair_embeddings(embs, labels):
    """
    Create Aelfric-Aelfric (positive) and Aelfric-Unknown (negative) pairs.
    Returns two lists of concatenated embeddings.
    """
    a_embs = []
    u_embs = []

    for i, emb in enumerate(embs):
        if labels[i] == 1:
            a_embs.append(emb)
        else:
            u_embs.append(emb)

    a_sample = random.sample(a_embs, k=min(250, len(a_embs))) ### Eventually the samples need to be controlled better
    u_sample = random.sample(u_embs, k=min(250, len(u_embs)))

    aa_pairs = []
    au_pairs = []

    # Aelfric–Aelfric pairs (positive)
    for emb_a in a_sample:
        for emb_b in a_sample:
            aa_pairs.append(np.concatenate([emb_a, emb_b]))

    # Aelfric–Unknown pairs (negative)
    for emb_a in a_sample:
        for emb_b in u_sample:
            au_pairs.append(np.concatenate([emb_a, emb_b]))

    return aa_pairs, au_pairs


def prepare_data():
    xTrainOld, xTest, yTrainOld, yTest, _, idTest, _ = loadData("data/sorted_embeddings/Aelfric/chunks", "data/sorted_embeddings/Unknown/chunks", 0.2, 42)
    
    print("Loaded embeddings...")
    print(f"Train Embeddings: {xTrainOld} ({len(xTrainOld)})")
    print(f"Train Labels: {yTrainOld} ({len(yTrainOld)})")
    print(f"Test Embeddings: {xTest} ({len(xTest)})")
    print(f"Test Labels: {yTest} ({len(yTest)})")

    # Separate out Aelfric training samples for pairing during prediction
    aelfric_train_sample = [xTrainOld[i] for i in range(len(xTrainOld)) if yTrainOld[i] == 1]

    aa_pairs, au_pairs = pair_embeddings(xTrainOld, yTrainOld)

    # --- Combine and label training pairs ---
    X_aa = np.stack(aa_pairs)
    X_au = np.stack(au_pairs)

    y_aa = np.ones(len(X_aa))   # +1 for Aelfric–Aelfric
    y_au = -np.ones(len(X_au))  # -1 for Aelfric–Unknown

    xTrain = np.vstack([X_aa, X_au])
    yTrain = np.concatenate([y_aa, y_au])

    return np.array(xTrain), np.array(yTrain), np.array(xTest), np.array(yTest), aelfric_train_sample


'''
def getIdAccuracy(model, device, data, moreprints):
    textBuckets = {}

    model.eval()
    with torch.no_grad():
        for x, y, ids in data:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            preds = out.argmax(dim=1)             

            for id_value, y_val, pred in zip(ids, y.cpu(), preds.cpu()):
                y_val = y_val.item()
                pred = pred.item()

                if id_value not in textBuckets:
                    # store: [true_label, pred0_count, pred1_count]
                    textBuckets[id_value] = [y_val, 0, 0, []]

                textBuckets[id_value][1 if pred == 0 else 2] += 1
                textBuckets[id_value][3].append((y_val == pred) + (y_val/2))

    totalAccuracy = 0
    goodAccuracy = 0
    totalCount = 0

    textAccuracies = []

    matrix = []
    texts = []

    for key, count in textBuckets.items():
        true_label, pred0, pred1, listOfPred = count
        total = pred0 + pred1

        if true_label == 0:
            accuracy = pred0 / total * 100
        else:
            accuracy = pred1 / total * 100

        if accuracy >= 50:
            totalAccuracy += 1
        if accuracy == 100:
            goodAccuracy += 1
        totalCount += 1
        if(moreprints):
            print(f"{key} | value {true_label} | size {total} : {accuracy:.2f}%")

        matrix.append(listOfPred)
        texts.append(key)

        textAccuracies.append(accuracy)

    max_len = max(len(row) for row in matrix)

    matrix = np.array([
        row + [2] * (max_len - len(row))for row in matrix
    ]).T

    cmap = ListedColormap(["red","#FFAAAA","blue", "lightblue", "white"])
    
    plt.figure(figsize=(35,8))
    plt.imshow(matrix, cmap=cmap, aspect='auto')
    im = sns.heatmap(matrix, cbar=False, cmap=cmap, xticklabels=texts, linewidths=0.05, linecolor="white",) #, yticklabels=yAxis, xticklabels=yAxis)
    im.invert_yaxis()

    plt.title("List of Chunks")
    plt.xlabel("Texts")
    plt.ylabel("Chunks")
    plt.savefig(f"data/visualizations/chunkheatmap.png")

    print(f"Chunk Accuracy {totalAccuracy/totalCount*100:.2f}%")
    print(f"Good Accuracy {goodAccuracy/totalCount*100:.2f}%")

    return textAccuracies
'''


def train_linear_model(X_train, y_train, X_test, y_test, aelfric_train_sample, num_epochs=5, batch_size=32):
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)  # shape (N_total, 1)

    # Create per-sample weights: slightly downweight Unknown (-1)
    #weights = np.ones(len(y_train), dtype=np.float32)
    #weights[y_train == -1] = 0.9
    #weights_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(device)

    model = LinearBinaryModel(input_dim=X_train.shape[1]).to(device)

    criterion = nn.MSELoss() #reduction="none")  
    optimizer = torch.optim.Adam(model.parameters(), 5e-5)

    # Dataset with weights
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    in_acc_total = []
    in_acc_aelfric = []
    in_acc_unknown = []
    out_acc_total = []
    out_acc_aelfric = []
    out_acc_unknown = []

    for epoch in range(num_epochs):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            s = model(X_batch)  # predicted score
            loss = criterion(s, y_batch)
            loss.backward()
            optimizer.step()

        # After each epoch, save in-sample and out-of-sample accuracy
        in_predictions = predict_linear_model(model, X_train)
        in_acc_total.append(accuracy(y_train, in_predictions)*100)
        in_accs = class_accuracy(y_train, in_predictions)
        in_acc_aelfric.append(in_accs["Aelfric"]*100)
        in_acc_unknown.append(in_accs["Unknown"]*100)

        out_predictions = predict_linear_model(model, X_test, aelfric_train_sample)
        out_acc_total.append(accuracy(y_test, out_predictions)*100)
        out_accs = class_accuracy(y_test, out_predictions)
        out_acc_aelfric.append(out_accs["Aelfric"]*100)
        out_acc_unknown.append(out_accs["Unknown"]*100)

    return model, \
           {"Accuracy": in_acc_total, "Aelfric Accuracy": in_acc_aelfric, "Unknown Accuracy": in_acc_unknown}, \
           {"Accuracy": out_acc_total, "Aelfric Accuracy": out_acc_aelfric, "Unknown Accuracy": out_acc_unknown}


def predict_linear_model(model, x_new, aelfric_train_sample=None):
    """
    x_new: np.array of shape (1024,) or (N, 1024)
    returns: list of +1 (Aelfric) or -1 (Unknown)
    """
    if aelfric_train_sample is None:
        # Standard case (already 2048-dim)
        if x_new.ndim == 1:
            x_new = x_new[np.newaxis, :]
        with torch.no_grad():
            X_tensor = torch.tensor(x_new, dtype=torch.float32).to(device)
            s = model(X_tensor).squeeze().detach().cpu().numpy()
        return [1 if score > 0 else -1 for score in s]
    else:
        # Pair each single 1024-dim test embedding with all Aelfric train embeddings
        preds = []
        for test_emb in x_new:
            pred = predict_single_embedding(model, test_emb, aelfric_train_sample)
            preds.append(pred)
        return preds
    

def predict_single_embedding(model, test_emb, aelfric_train_sample):
    """
    test_emb: np.array of shape (1024,)
    aelfric_train_sample: list of np.arrays of shape (1024,)
    Returns: single +1/-1 prediction for the test embedding
    """

    # Sample fewer Aelfric embeddings for speed
    aelfrics = random.sample(aelfric_train_sample, k=min(len(aelfric_train_sample), 50))

    pairs = [np.concatenate([a, test_emb]) for a in aelfrics]  # shape (N_a, 2048)
    X_tensor = torch.tensor(np.stack(pairs), dtype=torch.float32).to(device)

    with torch.no_grad():
        scores = model(X_tensor).squeeze().detach().cpu().numpy()  # one score per pair

    # Aggregate:
    avg_score = np.mean(scores)
    return 1 if avg_score > 0 else -1
    

def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


def class_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Check if class has any samples before computing mean
    aelfric_mask = y_true == 1
    if np.any(aelfric_mask):
        acc_aelfric = np.mean(y_pred[aelfric_mask] == 1)
    else:
        acc_aelfric = np.nan
    
    unknown_mask = y_true == -1
    if np.any(unknown_mask):
        acc_unknown = np.mean(y_pred[unknown_mask] == -1)
    else:
        acc_unknown = np.nan
    
    return {"Aelfric": acc_aelfric, "Unknown": acc_unknown}


def error(accs : dict) -> dict: 
    errors = {}
    errors["Error"] = 100 - np.array(accs["Accuracy"])
    errors["Aelfric Error"] = 100 - np.array(accs["Aelfric Accuracy"])
    errors["Unknown Error"] = 100 - np.array(accs["Unknown Accuracy"])
    return errors


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}") 
    

def main():
    logger = setup_logging()

    # Train the model
    X_train, y_train, X_test, y_test, aelfric_train_sample = prepare_data()
    model, in_accs, out_accs = train_linear_model(X_train, y_train, X_test, y_test, aelfric_train_sample)
    logger.info("Training complete.")

    # Find and print accuracies
    y_pred = predict_linear_model(model, X_test, aelfric_train_sample)

    acc_overall = accuracy(y_test, y_pred)
    class_acc = class_accuracy(y_test, y_pred)

    logger.info(f"Overall test accuracy: {acc_overall*100:.2f}%")
    logger.info(f"Aelfric accuracy: {class_acc['Aelfric']*100:.2f}%")
    logger.info(f"Unknown accuracy: {class_acc['Unknown']*100:.2f}%")

    # Save the model with descriptive filename
    acc_str = f"{round(acc_overall*100)}_{round(class_acc['Aelfric']*100)}_{round(class_acc['Unknown']*100)}"
    timestamp_str = datetime.now().strftime("%m_%d_%Y %H-%M")
    model_filename = f"models/linear_model ({acc_str}) ({timestamp_str}).pth" 
    save_model(model, model_filename)

    # Visualizations
    in_errors = error(in_accs)
    out_errors = error(out_accs)

    visualization(model, in_errors, "In Sample Error", "inSample", "")
    visualization(model, out_errors, "Out of Sample Error", "outSample", "")
    visualization(model, {"In Sample Error": in_errors["Error"], "Out of Sample Error": out_errors["Error"]}, "Both Error", "bothSample", "")

    # Current main problem with the code is that the embeddings might not be paired in the load_data function


if __name__ == '__main__':
    #res = generate_umap()

    #print(res)

    main()