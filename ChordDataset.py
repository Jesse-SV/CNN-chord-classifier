from imports import *

class ChordDataset(Dataset):
    def __init__(self, csv_path, dir_path, split, cut=30000):
        self.data = []
        self.labels = []
        self.dir_path = dir_path
        self.cut = cut
        self.split = split

        self.df = pd.read_csv(csv_path)

        for idx, row in self.df.iterrows():
            if row['split'] != split:
                continue

            chord_type = row['mode']
            file_name = row['name'] + ".wav"
            file_path = os.path.join(dir_path, chord_type, file_name)

            if not os.path.isfile(file_path):
                print(f"Warning: File not found: {file_path}")
                continue

            if chord_type not in CLASSES:
                # print(f"Warning: Class '{chord_type}' not in CLASSES")
                continue

            label = CLASSES.index(chord_type)
            self.data.append(file_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        label = self.labels[idx]
        waveform, sample_rate = torchaudio.load(path)  

        waveform = waveform[:, :self.cut]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True) 

        spectrogram_transform = torchaudio.transforms.Spectrogram()
        specgram = spectrogram_transform(waveform)

        return specgram, label