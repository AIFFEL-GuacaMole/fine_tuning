import os
import logging as log
import torch
from torch_geometric.data import DataLoader
from torch_geometric.nn import SchNet
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc
from tdc.benchmark_group import admet_group
from tdc.single_pred import Tox
from tqdm import tqdm
from torch_geometric.data import Data
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
import wandb

def setup_logging():
    log.basicConfig(level=log.INFO)
    log.getLogger("torch_geometric").setLevel(log.WARNING)

def _attempt_chirality_flip(mol):
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    if not chiral_centers:
        return Chem.AddHs(mol)

    for (atom_idx, chirality) in chiral_centers:
        mol_copy = Chem.Mol(mol)
        atom = mol_copy.GetAtomWithIdx(atom_idx)
        atom.InvertChirality()
        molH = Chem.AddHs(mol_copy)

        if AllChem.EmbedMolecule(molH, maxAttempts=10) != -1:
            return molH

    return Chem.AddHs(mol)

def save_molecule_data(file_path, molecules):
    """Molecule 데이터를 파일에 저장"""
    with open(file_path, 'wb') as f:
        pickle.dump(molecules, f)

def load_molecule_data(file_path):
    """파일에서 Molecule 데이터를 로드"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def generate_3D_coordinates_with_cache(df, cache_path):
    """
    Generate 3D coordinates for molecules, with caching support.
    """
    if os.path.exists(cache_path):
        log.info(f"Loading cached molecule data from {cache_path}...")
        return load_molecule_data(cache_path)

    log.info(f"Generating coordinates for {len(df)} molecules...")
    molecules = []

    for smiles in df['Drug']:
        mol = Chem.MolFromSmiles(smiles)
        y = df.loc[df['Drug'] == smiles, 'Y'].values[0]  # 정답 레이블

        if mol is not None:
            mol = Chem.AddHs(mol)
            try:
                result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                if result == -1:
                    mol = _attempt_chirality_flip(mol)
                    result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                if result == -1:
                    AllChem.Compute2DCoords(mol)

                conf = mol.GetConformer()
                coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
                atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

                molecules.append(Data(
                    pos=torch.tensor(coords, dtype=torch.float),
                    z=torch.tensor(atomic_numbers, dtype=torch.long),
                    y=torch.tensor([y], dtype=torch.float)
                ))
                continue
            except Exception as e:
                log.error(f"Failed to process SMILES: {smiles}. Error: {str(e)}")

        log.warning(f"Invalid or unprocessable SMILES: {smiles}. Adding dummy data.")
        molecules.append(Data(
            pos=torch.zeros((1, 3), dtype=torch.float),  # 더미 좌표
            z=torch.tensor([0], dtype=torch.long),       # 더미 원자 번호
            y=torch.tensor([y], dtype=torch.float)       # 원본 레이블 유지
        ))

    # 캐시에 저장
    save_molecule_data(cache_path, molecules)
    log.info(f"Molecule data cached at {cache_path}.")
    return molecules

def train_schnet(model, train_loader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.z, data.pos, data.batch).squeeze()
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_schnet(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    probabilities = []
    targets = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data.z, data.pos, data.batch).squeeze()
            if output.shape[0] != data.y.shape[0]:
                log.error(f"Mismatch in output and true size. Output: {output.shape[0]}, True: {data.y.shape[0]}")
                raise ValueError("Output and true size mismatch.")
            loss = criterion(output, data.y)
            total_loss += loss.item()

            probabilities.extend(torch.sigmoid(output).cpu().numpy())
            targets.extend(data.y.cpu().numpy())

    precision, recall, _ = precision_recall_curve(targets, probabilities)
    auprc = auc(recall, precision)

    log.debug(f"Batch output shape: {output.shape}, Batch y_true shape: {data.y.shape}")

    return total_loss / len(loader), auprc, probabilities

def train_schnet_with_auprc(model, train_loader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    probabilities = []
    targets = []

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.z, data.pos, data.batch).squeeze()
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # detach()를 사용하여 그래디언트를 분리
        probabilities.extend(torch.sigmoid(output).detach().cpu().numpy())
        targets.extend(data.y.cpu().numpy())

    # AUPRC 계산
    precision, recall, _ = precision_recall_curve(targets, probabilities)
    train_auprc = auc(recall, precision)

    return total_loss / len(train_loader), train_auprc

def main():
    setup_logging()

    # Initialize Weights and Biases
    wandb.init(project="SchNet_ADMET", config={
        'learning_rate': 0.001,
        'batch_size': 256,
        'hidden_channels': 512,
        'num_filters': 256,
        'epochs': 100,
        'early_stopping_patience': 10
    })
    config = wandb.config

    os.makedirs("cache", exist_ok=True)

    datasets = ['CYP2C9_Veith']
    checkpoint_path = "./schnet_20M.pt"
    predictions_list = []

    for seed in [1, 2, 3, 4, 5]:
        for dataset_name in datasets:
            log.info(f"Processing dataset: {dataset_name} with seed {seed}")

            group = admet_group(path='data/')
            benchmark = group.get(dataset_name)
            name = benchmark['name']
            train_val, test = benchmark['train_val'], benchmark['test']
            train, valid = group.get_train_valid_split(benchmark=name, split_type='scaffold', seed=seed)

            log.info(f"Train size: {len(train)}, Valid size: {len(valid)}, Test size: {len(test)}")
            log.info(f"Unique samples in train: {len(train['Drug'].unique())}")
            log.info(f"Unique samples in test: {len(test['Drug'].unique())}")

            train_cache = f"cache/{dataset_name}_train.pkl"
            valid_cache = f"cache/{dataset_name}_valid.pkl"
            test_cache = f"cache/{dataset_name}_test.pkl"

            train_data_list = generate_3D_coordinates_with_cache(train, train_cache)
            valid_data_list = generate_3D_coordinates_with_cache(valid, valid_cache)
            test_data_list = generate_3D_coordinates_with_cache(test, test_cache)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            train_loader = DataLoader(train_data_list, batch_size=config.batch_size, shuffle=True)
            valid_loader = DataLoader(valid_data_list, batch_size=config.batch_size, shuffle=False)
            test_loader = DataLoader(test_data_list, batch_size=config.batch_size, shuffle=False)

            model = SchNet(
                hidden_channels=config.hidden_channels,
                num_filters=config.num_filters,
                num_interactions=8,
                cutoff=15.0,
                num_gaussians=75
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
            criterion = torch.nn.BCEWithLogitsLoss()

            best_valid_loss = float('inf')
            patience_counter = 0

            for epoch in tqdm(range(config.epochs), desc=f"Training {dataset_name} with seed {seed}"):
                train_loss, train_auprc = train_schnet_with_auprc(model, train_loader, optimizer, device, criterion)
                valid_loss, valid_auprc, _ = evaluate_schnet(model, valid_loader, device, criterion)

                log.info(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train AUPRC: {train_auprc:.4f}, Valid AUPRC: {valid_auprc:.4f}")

                # Log metrics to W&B
                wandb.log({
                    "Train Loss": train_loss,
                    "Train AUPRC": train_auprc,
                    "Validation Loss": valid_loss,
                    "Valid AUPRC": valid_auprc
                })

                scheduler.step(valid_loss)

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), checkpoint_path)
                    log.info(f"best model epoch{epoch} saved.")
                else:
                    patience_counter += 1
                    if patience_counter >= config['early_stopping_patience']:
                        log.info("Early stopping triggered.")
                        break

            test_loss, test_auprc, y_pred = evaluate_schnet(model, test_loader, device, criterion)
            log.info(f"Test AUPRC for {dataset_name} with seed {seed}: {test_auprc:.4f}")

            # Log test metrics to W&B
            wandb.log({
                    "Train Loss": train_loss,
                    "Train AUPRC": train_auprc,
                    "Validation Loss": valid_loss,
                    "Valid AUPRC": valid_auprc
                })

            predictions = {dataset_name: y_pred}
            predictions_list.append(predictions)

    group = admet_group(path='data/')
    results = group.evaluate_many(predictions_list)
    log.info(f"Final Results: {results}")

    # Save final results to W&B
    wandb.log({"Final Results": results})

if __name__ == '__main__':
    main()
