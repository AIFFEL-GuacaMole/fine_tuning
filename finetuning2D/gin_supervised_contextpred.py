import torch
from torch.utils.data import Dataset, DataLoader
from molfeat.trans.pretrained import PretrainedDGLTransformer
from sklearn.metrics import precision_recall_curve, auc, accuracy_score
from tdc.single_pred import ADME
from dgllife.model import load_pretrained
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from itertools import product


# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # prevents nans when probability is 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# Load and preprocess data
def load_data():
    split = 'scaffold'
    data = ADME(name='CYP2C9_Veith')
    split_data = data.get_split(method=split)
    return split_data['train'], split_data['valid'], split_data['test']

def prepare_features(smiles_data):
    transformer = PretrainedDGLTransformer(kind='gin_supervised_contextpred', dtype=float)
    features = transformer(smiles_data)
    return features

# Custom Dataset Class
class SMILESFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        node_features = torch.tensor(feature, dtype=torch.float)
        return node_features, torch.tensor(label, dtype=torch.float)

def create_dataloaders(train_data, valid_data, test_data, batch_size):
    train_dataset = SMILESFeatureDataset(
        prepare_features(train_data['Drug'].tolist()), train_data['Y'].tolist()
    )
    valid_dataset = SMILESFeatureDataset(
        prepare_features(valid_data['Drug'].tolist()), valid_data['Y'].tolist()
    )
    test_dataset = SMILESFeatureDataset(
        prepare_features(test_data['Drug'].tolist()), test_data['Y'].tolist()
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


# Fine-Tuning 모델 정의
class FineTuningModel(nn.Module):
    def __init__(self, pretrained_model, input_dim, hidden_dim=128, num_layers=3, dropout=0.5):
        super(FineTuningModel, self).__init__()
        self.pretrained = pretrained_model
        for param in self.pretrained.parameters():
            param.requires_grad = False  # Pretrained 모델 freeze

        # MLP 레이어 정의
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))  # 첫 번째 레이어
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))  # 드롭아웃 추가

        # 추가적인 MLP 레이어
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # 마지막 레이어
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())  # 이진 분류를 위한 Sigmoid 출력

        self.mlp = nn.Sequential(*layers)  # MLP 네트워크를 시퀀스 형태로 결합

    def forward(self, node_feats):
        #embeddings = self.pretrained(node_feats)
        x = self.mlp(node_feats)  # MLP를 통한 전방향 계산
        return x



# AUPRC Calculation
def compute_auprc(data_loader, model, device):
    y_true, y_pred = [], []
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)


# Accuracy Calculation
def compute_accuracy(data_loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


pretrained_model = load_pretrained('gin_supervised_contextpred')

# Training function
def train(config= None):
    with wandb.init(config=config):
        config = wandb.config
        
        train_data, valid_data, test_data = load_data()
        batch_size = 32
        train_loader, valid_loader, test_loader = create_dataloaders(train_data, valid_data, test_data, batch_size)
        input_dim = prepare_features(train_data['Drug'].tolist()).shape[1]
        
        print(f"Train Data Size: {len(train_data)}")
        print(f"Validation Data Size: {len(valid_data)}")
        print(f"Test Data Size: {len(test_data)}")

        model = FineTuningModel(
            pretrained_model,    
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout
        ).to("cuda")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = FocalLoss() 
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        
        best_val_auprc = 0
        patience = 10
        epochs_without_improvement = 0

        for epoch in range(200):
            model.train()
            train_loss = 0
            for batch_features, batch_labels in train_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                outputs = model(batch_features).squeeze()
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_auprc = compute_auprc(train_loader, model, device)
            train_accuracy = compute_accuracy(train_loader, model, device)

            model.eval()
            val_loss = 0
            for batch_features, batch_labels in valid_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features).squeeze()
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(valid_loader)
            val_auprc = compute_auprc(valid_loader, model, device)
            val_accuracy = compute_accuracy(valid_loader, model, device)


            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_auprc": train_auprc,
                "train_accuracy": train_accuracy,
                "val_loss": avg_val_loss,
                "val_auprc": val_auprc,
                "val_accuracy": val_accuracy
            })

            print(
            f"Epoch {epoch + 1}:\n"
            f"  Train: Loss = {avg_train_loss:.4f}, AUPRC = {train_auprc:.4f}, Accuracy = {train_accuracy:.4f}\n"
            f"  Val:   Loss = {avg_val_loss:.4f}, AUPRC = {val_auprc:.4f}, Accuracy = {val_accuracy:.4f}"
            )


            # Save best model
            if val_auprc > best_val_auprc:
                best_val_auprc = val_auprc
                torch.save(model.state_dict(), "0118_best_model_gin.pth")
                print(f"New best model saved with AUPRC: {best_val_auprc:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            scheduler.step(val_auprc)

            # Early stopping
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break



# Main function
def main():
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'val_auprc',
            'goal': 'maximize'
        },
        'parameters': {
            'hidden_dim': {'values': [64, 128, 256]},
            'num_layers': {'values': [2, 3, 4]},
            'dropout': {'values': [0.3, 0.5, 0.7]},
            'learning_rate': {'values': [1e-3, 1e-4]},
            'batch_size': {'values': [16, 32]},
            'epochs': {'value': 200},
            'activation_function': {'values': ['relu', 'tanh']},
            'optimizer': {'values': ['adam', 'adamw']},
        }
    }

    
    sweep_id = wandb.sweep(sweep_config, project="CYP2C9_Veith_gin_supervised_contextpred_finetuning_sweep")


    
    wandb.agent(sweep_id, function=train)



if __name__ == "__main__":
    main()



