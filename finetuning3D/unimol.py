import logging
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
from unimol_tools import UniMolRepr
import os
import wandb

# unimol_tools 라이브러리의 INFO 로그 억제
logging.getLogger("unimol_tools").setLevel(logging.WARNING)

# Define sweep configuration
sweep_config = {
    "method": "grid",
    "parameters": {
        "batch_size": {
            "values": [32, 64, 128]
        },
        "learning_rate": {
            "values": [1e-2, 5e-3, 1e-3]
        },
        "hidden_dim": {
            "values": [128, 256, 512]
        },
        "dropout_rate": {
            "values": [0.1, 0.2, 0.3]
        },
        "epochs": {
            "values": [50, 100]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="unimol-mlp-auprc")

class UniMol_3D(nn.Module):

    def __init__(self, 
                 transformer_model='unimolv1', 
                 model_size='84m', 
                 num_classes=2,
                 hidden_dim=256,  
                 dropout_rate=0.2,
                 max_atomic_len=None):   
        super(UniMol_3D, self).__init__()

        self.unimol = UniMolRepr(
            data_type='molecule', 
            remove_hs=False,
            model_name=transformer_model, 
            model_size=model_size
        )

        self.cls_embedding_dim = 512
        self.atomic_embedding_dim = 512

        self.max_atomic_len = max_atomic_len

        self.attention_layer = nn.MultiheadAttention(embed_dim=self.atomic_embedding_dim, num_heads=8, batch_first=True)

        self.dropout = nn.Dropout(dropout_rate)
        self.mlp_combined = nn.Sequential(
            nn.Linear(self.cls_embedding_dim + self.atomic_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, smiles_list):
        reprs = self.unimol.get_repr(smiles_list, return_atomic_reprs=True)
        cls_embeddings = torch.tensor(reprs['cls_repr'], dtype=torch.float32)

        atomic_emb_list = [torch.tensor(atom_repr, dtype=torch.float32) for atom_repr in reprs['atomic_reprs']]
        atomic_embeddings = pad_sequence(atomic_emb_list, batch_first=True)

        device = self.mlp_combined[0].weight.device
        cls_embeddings = cls_embeddings.to(device)
        atomic_embeddings = atomic_embeddings.to(device)

        if self.max_atomic_len:
            if atomic_embeddings.size(1) < self.max_atomic_len:
                padding = torch.zeros(
                    atomic_embeddings.size(0),
                    self.max_atomic_len - atomic_embeddings.size(1),
                    self.atomic_embedding_dim,
                    device=device
                )
                atomic_embeddings = torch.cat((atomic_embeddings, padding), dim=1)
            elif atomic_embeddings.size(1) > self.max_atomic_len:
                atomic_embeddings = atomic_embeddings[:, :self.max_atomic_len, :]

        attn_output, _ = self.attention_layer(atomic_embeddings, atomic_embeddings, atomic_embeddings)
        atomic_summary = attn_output.mean(dim=1)  

        combined_embeddings = torch.cat((cls_embeddings, atomic_summary), dim=1)

        x = self.dropout(combined_embeddings)
        logits = self.mlp_combined(x)

        return logits, cls_embeddings, atomic_summary

# CSV 파일에 feature column 추가
def save_features_to_csv(csv_path, model):
    # 기존 CSV 파일 로드
    df = pd.read_csv(csv_path)

    # Drug 열에서 SMILES 리스트 생성
    if 'Drug' not in df.columns:
        raise ValueError("The CSV file must contain a 'Drug' column with SMILES codes.")

    smiles_list = df['Drug'].tolist()  # Drug 열에서 SMILES 가져오기

    # UniMolRepr로 임베딩 생성
    reprs = model.unimol.get_repr(smiles_list, return_atomic_reprs=True)

    # cls_embedding_dim와 atomic_embedding_dim 저장
    df['feature_cls'] = [','.join(map(str, cls)) for cls in reprs['cls_repr']]
    df['feature_atomic'] = [','.join(map(str, atom.mean(axis=0))) for atom in reprs['atomic_reprs']]

    # 변경된 DataFrame 저장
    df.to_csv(csv_path, index=False)
    print(f"Features saved to {csv_path}")

# Custom Dataset
class MoleculeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# Load CSV and preprocess
def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # Parse features
    if 'cls_embedding_dim' in df.columns and 'atomic_embedding_dim' in df.columns:
        feature_cls = [list(map(float, cls.split(','))) for cls in df['cls_embedding_dim']]
        feature_atomic = [list(map(float, atom.split(','))) for atom in df['atomic_embedding_dim']]
    else:
        raise ValueError("The CSV file must contain 'cls_embedding_dim' and 'atomic_embedding_dim' columns.")

    # Combine features
    features = [cls + atom for cls, atom in zip(feature_cls, feature_atomic)]

    # Encode labels (assuming labels are in a column called 'y')
    if 'y' not in df.columns:
        raise ValueError("The CSV file must contain a 'y' column for training.")
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['y'])

    return features, labels, label_encoder

# Compute AUPRC
def compute_auprc(loader, model):
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.numpy())
    return average_precision_score(all_labels, all_probs)

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.CrossEntropyLoss()(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss

# Training function
def train_mlp(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Load data
        csv_path = "molecule_coordinates_cache.csv"
        features, labels, label_encoder = load_data(csv_path)
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Create datasets and dataloaders
        train_dataset = MoleculeDataset(X_train, y_train)
        val_dataset = MoleculeDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        # Model, loss, optimizer
        model = MLP(input_dim=1024, hidden_dim=config.hidden_dim, num_classes=2, dropout_rate=config.dropout_rate)
        criterion = FocalLoss(alpha=1, gamma=2)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

        best_val_auprc = 0
        patience = 10
        patience_counter = 0

        # Training loop
        for epoch in range(config.epochs):
            model.train()
            train_loss = 0
            for batch_features, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()

            # Compute AUPRC
            train_auprc = compute_auprc(train_loader, model)
            val_auprc = compute_auprc(val_loader, model)

            # Log metrics to WandB
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss / len(train_loader),
                "val_loss": val_loss / len(val_loader),
                "train_auprc": train_auprc,
                "val_auprc": val_auprc,
                "val_accuracy": 100 * correct / total,
            })

            print(f"Epoch {epoch + 1}/{config.epochs}, Train Loss: {train_loss / len(train_loader):.4f}, "
                  f"Val Loss: {val_loss / len(val_loader):.4f}, Train AUPRC: {train_auprc:.4f}, "
                  f"Val AUPRC: {val_auprc:.4f}, Val Accuracy: {100 * correct / total:.2f}%")

            # Save best model
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                torch.save(model.state_dict(), "best_model_unimol.pth")
                print(f"New best model saved with AUPRC: {best_val_auprc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            scheduler.step(val_auprc)

            # Early stopping
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    wandb.agent(sweep_id, function=train_mlp)