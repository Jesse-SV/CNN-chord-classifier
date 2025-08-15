from imports import *

ROOT_CLASSES = ['A', 'Bb', 'B', 'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab']

class RootDataset(Dataset):
    def __init__(self, csv_path, dir_path, split='train', cut=30000):
        self.data = []
        self.labels = []
        self.cut = cut
        self.spec = torchaudio.transforms.Spectrogram()

        df = pd.read_csv(csv_path)
        for _, row in df[df['split'] == split].iterrows():
            root = row['root']
            if root not in ROOT_CLASSES:
                continue

            path = os.path.join(dir_path, row['mode'], f"{row['name']}.wav")
            if not os.path.isfile(path):
                continue

            self.data.append(path)
            self.labels.append(ROOT_CLASSES.index(root))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform, _ = torchaudio.load(self.data[idx])
        waveform = waveform.mean(dim=0, keepdim=True)[:, :self.cut]

        spec = self.spec(waveform)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return spec, label

