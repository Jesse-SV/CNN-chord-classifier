import os
import re
import time
import torch
import seaborn as sns
import torchaudio
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F



CLASSES = ['aug', 'aug6', 'dim', 'dim7', 'maj', 'maj2', 'maj3', 'maj6', 'maj7', 'maj7_2',
           'min', 'min2', 'min3', 'min6', 'min7', 'min7b5', 'perf4', 'perf5', 'seventh',
           'sixth', 'sus2', 'sus4', 'tritone']


# CLASSES = ['maj', 'min', 'aug', 'dim', 'sus2', 'sus4', 'dim7', 'maj7', 'min7', 'min7b5', 'seventh', 'sixth']

ROOT_CLASSES = ['A', 'Bb', 'B', 'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab']

def show_dataset(dataset, classes):
    print("Exibindo")
    for i in range(20):
        data, label = dataset[i]
        data_name = classes[label]
        print(f"Amostra {i+1}:")
        print(f" - Dados (shape): {data.shape}")
        print(f" - Label (índice): {label}")
        print(f" - Classe : {data_name}")

def evaluate(model, dataloader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device).float(), label.to(device).long()
            outputs = model(data)
            loss = criterion(outputs, label)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, avg_loss

'''

def model_exists(dir, batch_size, lr, dropout, epochs):
    pattern = re.compile(
        r"acc=\d+\.\d+_loss=\d+\.\d+_batch=(\d+)_lr=([0-9.]+)_drop=([0-9.]+)_epochs=(\d+)\.pt"
    )
    for filename in os.listdir(dir):
        match = pattern.match(filename)
        if match:
            b, l, d, e = match.groups()
            if int(b) == batch_size and float(l) == lr and float(d) == dropout and int(e) == epochs:
                return True
    return False
'''