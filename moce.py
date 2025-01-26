import torch 
import torch.nn as nn 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tdc.single_pred import ADME
from dgllife.model import load_pretrained
from molfeat.trans.pretrained import PretrainedDGLTransformer
from unimol_tools import UniMolRepr
from torch.nn import BCELoss
from torch.optim import Adam
from tdc.benchmark_group import admet_group
from sklearn.metrics import precision_recall_curve, auc
from models.moe import MoCE
import pandas as pd
import numpy as np
import os

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

def extract_chemberta_hidden_states(chemberta_inputs, model):
    with torch.no_grad():
        outputs = model(**chemberta_inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # 마지막 레이어의 hidden states
        cls_token_embedding = hidden_states[:, 0, :]  # CLS 토큰만 추출
    return cls_token_embedding

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
            chemberta_inputs = tokenizer(batch_smiles, padding=True, truncation=True, return_tensors="pt").to(device)
            chemberta_hidden_states = extract_chemberta_hidden_states(chemberta_inputs, chemberta_model)
            chemberta_features.append(chemberta_hidden_states)
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

    labels = torch.tensor(df_3d["y"].values, dtype=torch.float32).to(device).unsqueeze(-1)

    return chemberta_features, gin_features, unimol_features, labels


class MoCEForMultiModel(MoCE):
    def __init__(self, input_sizes, output_size, hidden_size, num_experts=3, dropout=0.5, k=None, kt=None):
        k = k if k is not None else num_experts
        kt = kt if kt is not None else num_experts
        super().__init__(hidden_size, output_size, num_experts, hidden_size, dropout, k, kt)

        # 각 feature 크기를 동일하게 맞추기 위한 MLP Projection Layer
        self.chemberta_mlp = nn.Sequential(
            nn.Linear(input_sizes[0], hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.gin_mlp = nn.Sequential(
            nn.Linear(input_sizes[1], hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.unimol_mlp = nn.Sequential(
            nn.Linear(input_sizes[2], hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, chemberta_output, gin_output, unimol_output,train_labels= None):

        # 데이터 타입 강제 변환
        chemberta_output = chemberta_output.float()
        gin_output = gin_output.float()
        unimol_output = unimol_output.float()

        # 텐서 크기 맞추기
        min_batch_size = min(chemberta_output.size(0), gin_output.size(0), unimol_output.size(0))
        chemberta_output = chemberta_output[:min_batch_size]
        gin_output = gin_output[:min_batch_size]
        unimol_output = unimol_output[:min_batch_size]
        if train_labels is not None:
            train_labels = train_labels[:min_batch_size]

        # MLP Projection Layer를 통해 각 feature 크기 맞추기
        chemberta_projected = self.chemberta_mlp(chemberta_output)  # (batch_size, hidden_size)
        gin_projected = self.gin_mlp(gin_output)                    # (batch_size, hidden_size)
        unimol_projected = self.unimol_mlp(unimol_output)           # (batch_size, hidden_size)

        # MoCE 로직 수행
        gates, _, _ = self.noisy_top_k_gating(chemberta_projected, self.training)
        combined_result = torch.sum(
            torch.stack([chemberta_projected, gin_projected, unimol_projected], dim=1) * gates.unsqueeze(-1), dim=1
        )
        # Output Layer를 통해 최종 출력 계산
        output = self.output_layer(combined_result)  # (batch_size, 1)

        if train_labels is not None:
            return torch.sigmoid(output), train_labels  # train_labels 반환
        else:
            return torch.sigmoid(output)  # 테스트 시 train_labels 없이 반환



def compute_auprc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

if __name__ == "__main__":
    admet_groups = admet_group(path='data/')
    benchmarks = admet_groups.get('CYP2C9_Veith')
    train_val, test = benchmarks['train_val'], benchmarks['test']

    tokenizer, chemberta_model = load_chemberta_model()
    gin_model = load_gin_model()
    unimol_model = load_unimol_model()

    batch_size = 32
    learning_rate = 1e-4
    input_sizes = [384, 300, 1024]  # Example input sizes

    train_chemberta, train_gin, train_unimol, train_labels = load_or_generate_features(
        "./molecule_coordinates_cache.csv", train_val, batch_size, gin_model, tokenizer, chemberta_model
    )

    test_chemberta, test_gin, test_unimol, test_labels = load_or_generate_features(
        "./molecule_coordinates_cache.csv", test, batch_size, gin_model, tokenizer, chemberta_model
    )

    moce = MoCEForMultiModel(input_sizes, output_size=1, hidden_size=512, num_experts=3).to(device)
    optimizer = Adam(moce.parameters(), lr=learning_rate)
    criterion = BCELoss()

    for epoch in range(2000):
        moce.train()
        optimizer.zero_grad()
        output, train_labels = moce(train_chemberta, train_gin, train_unimol, train_labels)
        loss = criterion(output, train_labels)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    moce.eval()
    with torch.no_grad():
        # 텐서 크기 맞추기
        min_batch_size = min(test_chemberta.size(0), test_gin.size(0), test_unimol.size(0), test_labels.size(0))
        test_chemberta = test_chemberta[:min_batch_size]
        test_gin = test_gin[:min_batch_size]
        test_unimol = test_unimol[:min_batch_size]
        test_labels = test_labels[:min_batch_size]

        test_output = moce(test_chemberta, test_gin, test_unimol)
        auprc = compute_auprc(test_labels.cpu().numpy(), test_output.cpu().numpy())
        print(f"Test AUPRC: {auprc:.4f}")

