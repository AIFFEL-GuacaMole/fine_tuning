import os
import torch
import torch.nn as nn
from torch.nn import BCELoss
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tdc.benchmark_group import admet_group
from dgllife.model import load_pretrained
from unimol_tools import UniMolRepr
import pandas as pd
import numpy as np
from molfeat.trans.pretrained import PretrainedDGLTransformer

def enable_gradient_checkpointing(model):
    if isinstance(model, AutoModelForSequenceClassification):
        model.roberta.encoder.gradient_checkpointing_enable()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8, 0)  

def load_chemberta_model():
    model_name = "DeepChem/ChemBERTa-77M-MLM"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    enable_gradient_checkpointing(model)
    return tokenizer, model.to(device)

def load_gin_model():
    model = load_pretrained('gin_supervised_contextpred')
    return model.to(device)

def load_unimol_model():
    model = UniMolRepr(data_type='molecule', remove_hs=False, model_name='unimolv1', model_size='84m')
    return model

def prepare_gin_features(smiles_data, gin_model):
    transformer = PretrainedDGLTransformer(kind='gin_supervised_contextpred')
    return transformer(smiles_data)

def load_or_generate_features(file_path_3d, data, batch_size, gin_model, tokenizer, chemberta_model):
    chemberta_cache_path = "./cache/chemberta_features.pt"
    gin_cache_path = "./cache/gin_features.pt"

    if os.path.exists(chemberta_cache_path) and os.path.exists(gin_cache_path):
        print("Loading cached features...")
        chemberta_features = torch.load(chemberta_cache_path).to(device)
        gin_features = torch.load(gin_cache_path).to(device)
    else:
        print("Generating features from scratch...")
        smiles = data["Drug"].tolist()

        chemberta_features = []
        for i in range(0, len(smiles), batch_size):
            batch_smiles = smiles[i:i + batch_size]
            chemberta_inputs = tokenizer(batch_smiles, padding=True, truncation=True, return_tensors="pt")
            chemberta_inputs = {key: val.to(device) for key, val in chemberta_inputs.items()}
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    batch_features = chemberta_model(**chemberta_inputs).logits
            chemberta_features.append(batch_features)
        chemberta_features = torch.cat(chemberta_features, dim=0)

        gin_features = []
        for i in range(0, len(smiles), batch_size):
            batch_smiles = smiles[i:i + batch_size]
            batch_features = prepare_gin_features(batch_smiles, gin_model)
            batch_features = torch.tensor(batch_features, dtype=torch.float32).to(device)
            gin_features.append(batch_features)
        gin_features = torch.cat(gin_features, dim=0)

        torch.save(chemberta_features, chemberta_cache_path)
        torch.save(gin_features, gin_cache_path)

    df_3d = pd.read_csv(file_path_3d)
    feature_cls = np.array([list(map(float, cls.split(','))) for cls in df_3d['cls_embedding_dim']])
    feature_atomic = np.array([list(map(float, atom.split(','))) for atom in df_3d['atomic_embedding_dim']])
    feature_cls = torch.tensor(feature_cls, dtype=torch.float32).to(device)
    feature_atomic = torch.tensor(feature_atomic, dtype=torch.float32).to(device)

    unimol_features = torch.cat((feature_cls, feature_atomic), dim=1)

    labels = torch.tensor(df_3d["y"].values, dtype=torch.float32).to(device)

    if 'Drug' not in data.columns or 'Y' not in data.columns:
        raise ValueError("Dataset must contain 'Drug' and 'Y' columns.")
    if data['Drug'].isnull().any() or data['Y'].isnull().any():
        data = data.dropna(subset=['Drug', 'Y'])

    return chemberta_features, gin_features, unimol_features, labels

class CrossAttention(nn.Module):
    def __init__(self, input_sizes, hidden_size, num_heads=8):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.projection = nn.Linear(sum(input_sizes), hidden_size)
        self.dropout = nn.Dropout(0.4)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, *inputs):
        combined = torch.cat(inputs, dim=-1)
        projected = self.projection(combined)
        attn_output, _ = self.attention(projected, projected, projected)
        return self.dropout(self.layer_norm(attn_output))

class MoEForMultiModel(nn.Module):
    def __init__(self, input_sizes, output_size, hidden_size, num_experts=3, dropout=0.4):
        super(MoEForMultiModel, self).__init__()
        self.cross_attention = CrossAttention(input_sizes, hidden_size)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=-1)
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 1024),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(1024, output_size)
            ) for _ in range(num_experts)
        ])

    def forward(self, chemberta_output, gin_output, unimol_output):
        combined_input = self.cross_attention(chemberta_output, gin_output, unimol_output)
        gate_outputs = self.gate(combined_input)
        expert_outputs = torch.stack([expert(combined_input) for expert in self.experts], dim=1)
        final_output = torch.sum(expert_outputs * gate_outputs.unsqueeze(-1), dim=1)
        return torch.sigmoid(final_output)

def main():
    admet_groups = admet_group(path='data/')
    benchmarks_auprc = [
        admet_groups.get('CYP2C9_Veith'),
        admet_groups.get('CYP2D6_Veith'),
        admet_groups.get('CYP3A4_Veith')
    ]
    benchmarks_auroc = [
        admet_groups.get('hERG'),
        admet_groups.get('AMES'),
        admet_groups.get('DILI')
    ]

    tokenizer, chemberta_model = load_chemberta_model()
    gin_model = load_gin_model()

    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 5e-5
    hidden_size = 1024

    for benchmarks, metric in [(benchmarks_auprc, 'AUPRC'), (benchmarks_auroc, 'AUROC')]:
        predictions = {}
        for benchmark in benchmarks:
            name = benchmark['name']
            train_val, test = benchmark['train_val'], benchmark['test']

            train_chemberta, train_gin, train_unimol, train_labels = load_or_generate_features(
                "./molecule_coordinates_cache.csv", train_val, batch_size, gin_model, tokenizer, chemberta_model
            )
            test_chemberta, test_gin, test_unimol, test_labels = load_or_generate_features(
                "./molecule_coordinates_cache.csv", test, batch_size, gin_model, tokenizer, chemberta_model
            )

            input_sizes = [train_chemberta.shape[1], train_gin.shape[1], train_unimol.shape[1]]
            output_size = 1
            moe = MoEForMultiModel(input_sizes, output_size, hidden_size, dropout=0.4).to(device)

            optimizer = Adam(moe.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = BCELoss()

            for epoch in range(150):
                moe.train()
                optimizer.zero_grad()
                train_output = moe(train_chemberta, train_gin, train_unimol).squeeze()
                train_loss = criterion(train_output, train_labels.float())
                train_loss.backward()
                optimizer.step()

            moe.eval()
            test_output_list = []
            for i in range(0, len(test), batch_size):
                batch_indices = slice(i, min(i + batch_size, len(test)))
                batch_chemberta = test_chemberta[batch_indices]
                batch_gin = test_gin[batch_indices]
                batch_unimol = test_unimol[batch_indices]
                with torch.no_grad():
                    batch_output = moe(batch_chemberta, batch_gin, batch_unimol).squeeze()
                test_output_list.append(batch_output)

            test_output = torch.cat(test_output_list, dim=0)
            y_pred = test_output.cpu().numpy()
            predictions[name] = y_pred

        results = admet_groups.evaluate(predictions)
        print(f"Evaluation Results for {metric}:", results)

if __name__ == "__main__":
    main()