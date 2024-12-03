import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

# Mapa de etiquetas de texto a valores numéricos
label_map = {
    'sin riesgo': 0,
    'riesgo medio': 1,
    'riesgo alto': 2
}

class MineriaDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = label_map[self.labels[item]]  # Convertir etiqueta de texto a valor numérico

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # [CLS] y [SEP]
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class RiesgoMineriaModel:
    def __init__(self, model_name="dccuchile/bert-base-spanish-wwm-uncased", num_labels=3):
        # Cargar el modelo y el tokenizador preentrenado
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained('modelo_entrenado', num_labels=num_labels)  # Cargar desde la carpeta donde guardaste el modelo
        self.model.eval()  # Establecer el modelo en modo evaluación (no entrenamiento)
    

    def train(self, train_loader, val_loader, epochs=3):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            for batch in tqdm(train_loader):
                self.optimizer.zero_grad()

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Loss: {avg_train_loss:.4f}")

            self.evaluate(val_loader)

    def evaluate(self, val_loader):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.eval()
        total_eval_accuracy = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                total_eval_accuracy += (predictions == labels).sum().item()

        avg_eval_accuracy = total_eval_accuracy / len(val_loader.dataset)
        print(f"Validation Accuracy: {avg_eval_accuracy:.4f}")

    def predict(self, text, umbral_riesgo_medio=0.6, umbral_riesgo_alto=0.8):
        self.model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            probabilidad = probabilities[0][prediction].item()

        reverse_label_map = {0: 'Sin Riesgo', 1: 'Riesgo Medio', 2: 'Riesgo Alto'}
        nivel_riesgo = reverse_label_map[prediction]
        alerta = {
            'texto': text,
            'nivel_riesgo': nivel_riesgo,
            'probabilidad': probabilidad,
            'generar_alerta': False,
            'mensaje_alerta': ''
        }

        if nivel_riesgo == 'Riesgo Medio' and probabilidad > umbral_riesgo_medio:
            alerta['generar_alerta'] = True
            alerta['mensaje_alerta'] = f"⚠️ ALERTA DE RIESGO MEDIO: Probabilidad {probabilidad:.2%}"

        elif nivel_riesgo == 'Riesgo Alto' and probabilidad > umbral_riesgo_alto:
            alerta['generar_alerta'] = True
            alerta['mensaje_alerta'] = f"🚨 ALERTA DE RIESGO ALTO: Probabilidad {probabilidad:.2%}"

        return alerta

if __name__ == "__main__":
    df = pd.read_csv('conversaciones_riesgo_mineria.csv')

    model = RiesgoMineriaModel()
    train_texts, val_texts, train_labels, val_labels = train_test_split(df['texto'].values, df['etiqueta'].values, test_size=0.1)
    train_dataset = MineriaDataset(train_texts, train_labels, model.tokenizer)
    val_dataset = MineriaDataset(val_texts, val_labels, model.tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    model.train(train_loader, val_loader)

    # Guardar los pesos del modelo entrenado
    model.model.save_pretrained('modelo_entrenado')
    model.tokenizer.save_pretrained('modelo_entrenado')