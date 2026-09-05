import numpy as np
import json
from pathlib import Path
import logging
import torch
import torch.nn as nn
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class LinearBinaryModel(nn.Module):
    #'''
    def __init__(self, input_dim=1024, hidden_dim=128):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)  # input → hidden
        self.relu = nn.ReLU()                            # non-linearity
        self.output = nn.Linear(hidden_dim, 1)          # hidden → output

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)   # scalar score
        return x
    #'''
    '''
    def __init__(self, input_dim=1024): 
        super().__init__() 
        self.linear = nn.Linear(input_dim, 1) # w + b 
        
    def forward(self, x): 
        return self.linear(x) # output s (can be positive or negative)
    '''


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


def visualization(model, lines, title, filename, otherData):
    modelString = str(model)

    # Count lines to estimate height needed 
    modelLines = len(modelString.splitlines())
    extraHeight = 0.1 + modelLines * 0.015  # adjust for text length
    
    # Create figure with dynamic bottom margin
    fig, ax = plt.subplots(figsize=(20, 5 + extraHeight * 10))
    fig.subplots_adjust(bottom=(extraHeight * 10) / ((extraHeight * 10) + 5))



    colors = plt.cm.jet(np.linspace(0.1,0.9,len(lines)))


    for i, (label, line) in enumerate(lines.items()):
        ax.plot(line, color=colors[i], label=label, alpha=0.7)
    
    
    ax.set_ylim(0, 101)
    ax.set_yticks(range(0, 105, 10))
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

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

'''
def load_embedding_pairs(results_dir_a: str, results_dir_b: str) -> list:
    embeddings_a = load_embeddings(results_dir_a)
    embeddings_b = load_embeddings(results_dir_b)

    # Convert dict values to lists once (so we don’t keep recreating them)
    values_a = list(embeddings_a.values())
    values_b = list(embeddings_b.values())

    # Randomly sample up to 200 values from each embedding set
    values_a = random.sample(list(embeddings_a.values()), k=min(500, len(values_a)))
    values_b = random.sample(list(embeddings_b.values()), k=min(500, len(values_b)))

    embedding_pairs = []
    for embedding_a in values_a:
        for embedding_b in values_b:
            # Concatenate the two 1024-dim embeddings → one 2048-dim vector
            embedding_pairs.append(np.concatenate([embedding_a, embedding_b]))

    return embedding_pairs
'''

def prepare_data():
    # Get aelfric and unknown embeddings
    aelfric_dict = load_embeddings("data/sorted_embeddings/Aelfric")
    unknown_dict = load_embeddings("data/sorted_embeddings/Unknown")

    # Stack the embeddings into matrices
    X_a = np.stack(list(aelfric_dict.values())) # shape (N_a, 1024)
    X_u = np.stack(list(unknown_dict.values())) # shape (N_u, 1024)

    y_a = np.ones(len(X_a)) # +1 for Aelfric
    y_u = -np.ones(len(X_u)) # -1 for Unknown

    X = np.vstack([X_a, X_u]) # shape (N_total, 1024)
    y = np.concatenate([y_a, y_u]) # shape (N_total,)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

def prepare_train_data():
    #'''
    # Get aelfric and unknown embeddings
    aelfric_dict = load_embeddings("data/sorted_embeddings/Aelfric/Train")
    unknown_dict = load_embeddings("data/sorted_embeddings/Unknown/Train")

    # Stack the embeddings into matrices
    X_a = np.stack(list(aelfric_dict.values())) # shape (N_a, 1024)
    X_u = np.stack(list(unknown_dict.values())) # shape (N_u, 1024)

    y_a = np.ones(len(X_a)) # +1 for Aelfric
    y_u = -np.ones(len(X_u)) # -1 for Unknown

    X_train = np.vstack([X_a, X_u]) # shape (N_total, 1024)
    y_train = np.concatenate([y_a, y_u]) # shape (N_total,)

    return X_train, y_train
    #'''
    '''
    # Get aelfric-aelfric and aelfric-unknown embedding pairs
    aelfric_aelfric_pairs = load_embedding_pairs("data/sorted_embeddings/Aelfric/Train", "data/sorted_embeddings/Aelfric/Train")
    aelfric_unknown_pairs = load_embedding_pairs("data/sorted_embeddings/Aelfric/Train", "data/sorted_embeddings/Unknown/Train")

    # Stack the embeddings into matrices
    X_a = np.stack(aelfric_aelfric_pairs) # shape (N_a, 2048)
    X_u = np.stack(aelfric_unknown_pairs) # shape (N_u, 2048)

    # Labels
    y_a = np.ones(len(X_a)) # +1 for Aelfric-Aelfric
    y_u = -np.ones(len(X_u)) # -1 for Aelfric-Unknown

    # Train data
    X_train = np.vstack([X_a, X_u]) # shape (N_total, 2048)
    y_train = np.concatenate([y_a, y_u]) # shape (N_total,)

    return X_train, y_train
    '''

def prepare_test_data():
    #'''
    aelfric_dict = load_embeddings("data/sorted_embeddings/Aelfric/Test")
    unknown_dict = load_embeddings("data/sorted_embeddings/Unknown/Test")

    X_a = np.stack(list(aelfric_dict.values())) # shape (N_a, 1024)
    X_u = np.stack(list(unknown_dict.values())) # shape (N_u, 1024)

    y_a = np.ones(len(X_a)) # +1 for Aelfric
    y_u = -np.ones(len(X_u)) # -1 for Unknown

    X_test = np.vstack([X_a, X_u]) # shape (N_total, 1024)
    y_test = np.concatenate([y_a, y_u]) # shape (N_total,)

    return X_test, y_test
    #'''
    '''
    aelfric_aelfric_pairs = load_embedding_pairs("data/sorted_embeddings/Aelfric/Test", "data/sorted_embeddings/Aelfric/Test")
    aelfri_unknown_pairs = locad_embedding_pairs("data/sorted_embeddings/Aelfric/Test", "data/sorted_embeddings/Unknown/Test")

    X_a = np.stack(aelfric_aelfric_pairs)
    X_u = np.stack(aelfric_unknown_pairs)

    y_a = np.ones(len(X_a))
    y_u = -np.ones(len(X_u))

    X_test = np.vstack([X_a, X_u])
    y_test = np.concatenate([y_a, y_u])

    return X_test, y_test
    '''


def train_linear_model(X_train, X_test, y_train, y_test, num_epochs=50, batch_size=32):
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) # shape (N_total, 1)

    model = LinearBinaryModel(input_dim=X_train.shape[1])
    
    criterion = nn.MSELoss() # MSE for -1/+1 labels
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

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
            optimizer.zero_grad()
            s = model(X_batch) # predicted score
            loss = criterion(s, y_batch)
            loss.backward()
            optimizer.step()

        # After each epoch, save the in-sample and out-of-sample accuracy in separate lists
        in_predictions = predict_linear_model(model, X_train)
        in_acc_total.append(accuracy(y_train, in_predictions)*100) # Total in-sample accuracy
        in_accs = class_accuracy(y_train, in_predictions) # Class-wise in-sample accuracy
        in_acc_aelfric.append(in_accs["Aelfric"]*100) # In-sample accuracy for Aelfric class
        in_acc_unknown.append(in_accs["Unknown"]*100) # In-sample accuracy for Unknown class

        out_predictions = predict_linear_model(model, X_test) # Total out-of-sample accuracy
        out_acc_total.append(accuracy(y_test, out_predictions)*100) # Total out-sample accuracy
        out_accs = class_accuracy(y_test, out_predictions) # Class-wise out-sample accuracy
        out_acc_aelfric.append(out_accs["Aelfric"]*100) # Out-sample accuracy for Aelfric class
        out_acc_unknown.append(out_accs["Unknown"]*100) # Out-sample accuracy for Unknown class

    return model, {"Accuracy": in_acc_total, "Aelfric Accuracy": in_acc_aelfric, "Unknown Accuracy": in_acc_unknown}, {"Accuracy": out_acc_total, "Aelfric Accuracy": out_acc_aelfric, "Unknown Accuracy": out_acc_unknown}


def predict_linear_model(model, x_new):
    """
    x_new: np.array of shape (1024,) or (N, 1024)
    returns: list of +1 (Aelfric) or -1 (Unknown)
    """
    if x_new.ndim == 1:
        x_new = x_new[np.newaxis, :]  # shape (1, 2048)
    
    with torch.no_grad():
        X_tensor = torch.tensor(x_new, dtype=torch.float32)
        s = model(X_tensor).squeeze().numpy()  # shape (N,)
        return [1 if score > 0 else -1 for score in s]
    

def accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


def class_accuracy(y_true, y_pred):
    """
    Compute accuracy for each class separately.
    
    y_true, y_pred: arrays/lists of +1/-1 labels
    Returns: dict with accuracy for each class
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    acc_aelfric = np.mean(y_pred[y_true == 1] == 1)
    acc_unknown = np.mean(y_pred[y_true == -1] == -1)
    
    return {"Aelfric": acc_aelfric, "Unknown": acc_unknown}
    

def main():
    logger = setup_logging()

    X_train, X_test, y_train, y_test = prepare_data()
    model, in_accs, out_accs = train_linear_model(X_train, X_test, y_train, y_test)
    logger.info("Training complete.")

    #print(in_accs)
    #print(out_accs)

    visualization(model, in_accs, "In Sample Accuracy", "inSample", "")
    visualization(model, out_accs, "Out of Sample Accuracy", "outSample", "")
    visualization(model, {"In Sample Accuracy": in_accs["Accuracy"], "Out of Sample Accuracy": out_accs["Accuracy"]}, "Both Accuracy", "bothSample", "")

    #'''
    y_pred = predict_linear_model(model, X_test)

    acc_overall = accuracy(y_test, y_pred)
    class_acc = class_accuracy(y_test, y_pred)

    logger.info(f"Overall test accuracy: {acc_overall*100:.2f}%")
    logger.info(f"Aelfric accuracy: {class_acc['Aelfric']*100:.2f}%")
    logger.info(f"Unknown accuracy: {class_acc['Unknown']*100:.2f}%")
    #'''

if __name__ == '__main__':
    #res = generate_umap()

    #print(res)

    main()