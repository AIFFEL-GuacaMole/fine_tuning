# Fine-Tuning for Molecular Property Prediction

이 레포지토리는 분자 특성 예측을 위한 다양한 표현 방식(1D, 2D, 3D)을 활용한 모델의 파인튜닝을 수행하는 데 필요한 코드와 출력을 포함하고 있습니다.

## 프로젝트 구조

```plaintext
fine_tuning/
├── data_transformation.py          # 데이터 변환 스크립트
├── finetuning1D/                   # 1D 분자 표현(SMILES) 기반 파인튜닝 관련 코드
│   ├── chemberta.py                # ChemBERTa 모델을 사용한 1D 표현 파인튜닝
├── finetuning2D/                   # 2D 분자 표현 기반 파인튜닝 관련 코드
│   ├── gin_supervised_contextpred.py # GIN(Graph Isomorphism Network) 모델을 활용한 파인튜닝
├── finetuning3D/                   # 3D 분자 구조 기반 파인튜닝 관련 코드
│   ├── SchNet_Fine_Tuning_Output/  # SchNet 모델 파인튜닝 결과물
│   │   ├── schnet_model/           # SchNet 모델 저장 디렉토리
│   │       ├── best_model.pth      # SchNet의 최적 파라미터가 저장된 파일
│   ├── fine_tuning3D.py            # 3D 구조 기반 파인튜닝 메인 스크립트
│   ├── unimol.py                   # Uni-Mol 모델을 사용한 파인튜닝 코드
```

## 주요 파일 설명

### **`data_transformation.py`**
- **역할:** 데이터 변환 스크립트입니다. 원시 분자 데이터를 모델 입력에 적합한 형식으로 변환합니다.

### **`finetuning1D/chemberta.py`**
- **역할:** ChemBERTa 모델을 활용하여 SMILES 형식의 1D 분자 표현 기반으로 파인튜닝을 수행합니다.

### **`finetuning2D/gin_supervised_contextpred.py`**
- **역할:** GIN (Graph Isomorphism Network)을 사용하여 2D 분자 그래프 기반의 특성을 학습합니다.

### **`finetuning3D/`**
- **역할:** 3D 분자 구조를 활용한 파인튜닝 코드를 포함합니다.

### **`SchNet_Fine_Tuning_Output/schnet_model/best_model.pth`**
- **역할:** SchNet 모델의 최적 파라미터가 저장된 파일입니다.

### **`fine_tuning3D.py`**
- **역할:** 3D 분자 구조를 학습하는 메인 스크립트입니다.

### **`unimol.py`**
- **역할:** Uni-Mol 모델을 사용한 3D 분자 구조 학습 스크립트입니다.


## Environment
Python 3.9

```

conda install pytorch-cluster -c pyg
conda install -c dglteam/label/th24_cu124 dgl

```


```
pip install torch_geometric
Pip install torch
pip install molfeat

pip install unimol_tools
pip install dgllife 
pip install numpy==1.23.5 pandas==1.3.3

pip install torchdata PyTDC rdkit-pypi transformers
pip install wandb

```

