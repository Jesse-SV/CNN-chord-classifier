import os
import torch
from torch.utils.data import Dataset

class PreprocessedRootDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        # Lista todos os arquivos .pt no diretório
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.pt')]
        self.files.sort()  # opcional: para ordem consistente

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        spec, label = torch.load(path)
        return spec, torch.tensor(label, dtype=torch.long)
