import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
SEQUENCE_LENGTH = 30
FEATURES = 42
EPOCHS = 30
BATCH_SIZE = 16

labels = sorted(os.listdir(DATA_DIR))
label_map = {label: i for i, label in enumerate(labels)}

X, y = [], []

for label in labels:
    for file in os.listdir(os.path.join(DATA_DIR, label)):
        X.append(np.load(os.path.join(DATA_DIR, label, file)))
        y.append(label_map[label])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(GestureDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(GestureDataset(X_test, y_test), batch_size=BATCH_SIZE)

class LSTMModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(FEATURES, 64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(len(labels)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)

        optimizer.zero_grad()
        output = model(Xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "sign_lstm_torch.pth")
np.save("labels.npy", labels)
