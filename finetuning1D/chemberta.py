import os
import wandb
from tdc.single_pred import ADME
import torch
import sklearn
import logging
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from sklearn.metrics import average_precision_score
import torch.nn.functional as F

wandb.init()

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# 데이터 로드
split = 'scaffold'
data = ADME(name='CYP2C9_Veith')
split_data = data.get_split(method=split)

train_df, valid_df, test_df = split_data['train'], split_data['valid'], split_data['test']

# set the logging directories
project_name = "chemberta_cyp"
output_path = './LLM_Fine_Tuning_Molecular_Properties_output'
model_name = 'chemberta-77M'

model_folder = os.path.join(output_path, model_name)

evaluation_folder = os.path.join(output_path, model_name + '_evaluation')
if not os.path.exists(evaluation_folder):
    os.makedirs(evaluation_folder)

# set the parameters
EPOCHS = 200
BATCH_SIZE = 256
patience = 10
learning_rate = 1e-5
manual_seed = 112
print(model_folder)

# configure Weights & Biases logging
wandb_kwargs = {'name' : model_name}

# Dataset class
def process_data(data, tokenizer, max_length=128):
    """Tokenizes the dataset."""
    texts = data['Drug'].tolist()
    labels = data['Y'].tolist()
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    return encodings, torch.tensor(labels)

class SMILESDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

def main():
    # Load model and tokenizer
    model_name = "DeepChem/ChemBERTa-77M-MLM"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Tokenize datasets
    train_encodings, train_labels = process_data(train_df, tokenizer)
    valid_encodings, valid_labels = process_data(valid_df, tokenizer)
    test_encodings, test_labels = process_data(test_df, tokenizer)

    train_dataset = SMILESDataset(train_encodings, train_labels)
    valid_dataset = SMILESDataset(valid_encodings, valid_labels)
    test_dataset = SMILESDataset(test_encodings, test_labels)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Optimizer and scheduler
    optimizer =  torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=EPOCHS * len(train_loader)
    )

    # Device setup
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Training function with early stopping
    def train(model, train_loader, valid_loader):
        model.train()
        best_ap = 0
        patience_counter = 0

        for epoch in range(EPOCHS):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                # Use Binary Cross Entropy Loss
                logits = outputs.logits
                labels = batch['labels'].float()  # BCE expects float labels
                loss = F.binary_cross_entropy_with_logits(logits[:, 1], labels)

                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")

            # Validation
            current_ap = evaluate(model, valid_loader)

            # Early stopping
            if current_ap > best_ap:
                best_ap = current_ap
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_ap': best_ap,
                    'hyperparameters': {
                        'EPOCHS': EPOCHS,
                        'BATCH_SIZE': BATCH_SIZE,
                        'patience': patience,
                        'learning_rate': learning_rate
                    }
                }, os.path.join(model_folder, f"best_model_E{EPOCHS}_B{BATCH_SIZE}_LR{learning_rate}_P{patience}.pt"))
                print("Model improved. Saving model.")
            else:
                patience_counter += 1
                print(f"No improvement. Patience counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Evaluation function
    def evaluate(model, loader):
        model.eval()
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                probs = torch.sigmoid(outputs.logits[:, 1])  # Probability of the positive class
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Verify probabilities are properly computed
        assert all(0 <= p <= 1 for p in all_probs), "Some probabilities are out of bounds!"

        average_precision = average_precision_score(all_labels, all_probs)
        print(f"Validation Average Precision (AP): {average_precision:.4f}")
        return average_precision

    # Train the model
    train(model, train_loader, valid_loader)

    # Test the model
    def test(model, loader):
        model.eval()
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                probs = torch.sigmoid(outputs.logits[:, 1])  # Probability of the positive class
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        average_precision = average_precision_score(all_labels, all_probs)
        print(f"Test Average Precision (AP): {average_precision:.4f}")
        return average_precision

    # Test results
    test_ap = test(model, test_loader)
    wandb.log({"test_average_precision": test_ap})
    wandb.finish()

if __name__ == "__main__":
    main()

