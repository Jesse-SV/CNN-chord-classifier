import os
import torch
import torchaudio
import pandas as pd
from tqdm import tqdm

ROOT_CLASSES = ['A', 'Bb', 'B', 'C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab']

class SpectrogramPreprocessor:
    def __init__(self, csv_path, audio_dir, output_dir, split='train', cut=30000):
        self.csv_path = csv_path
        self.audio_dir = audio_dir
        self.output_dir = output_dir
        self.split = split
        self.cut = cut
        self.spec_transform = torchaudio.transforms.Spectrogram()
        
        os.makedirs(self.output_dir, exist_ok=True)

    def process(self):
        df = pd.read_csv(self.csv_path)
        df = df[df['split'] == self.split]

        processed_count = 0
        for _, row in tqdm(df.iterrows(), total=len(df)):
            root = row['root']
            if root not in ROOT_CLASSES:
                continue

            audio_path = os.path.join(self.audio_dir, row['mode'], f"{row['name']}.wav")
            if not os.path.isfile(audio_path):
                print(f"Arquivo não encontrado: {audio_path}")
                continue

            waveform, sr = torchaudio.load(audio_path)
            # Convert stereo to mono
            waveform = waveform.mean(dim=0, keepdim=True)
            # Cut waveform length
            waveform = waveform[:, :self.cut]

            # Compute spectrogram
            spec = self.spec_transform(waveform)

            label = ROOT_CLASSES.index(root)
            save_path = os.path.join(self.output_dir, f"{row['name']}_{self.split}.pt")

            # Save tuple (spec, label)
            torch.save((spec, label), save_path)
            processed_count += 1

        print(f"Total de espectrogramas processados: {processed_count}")

if __name__ == "__main__":
    csv_path = "dataset/chords.csv"
    audio_dir = "dataset/chords/"
    output_dir_train = "spectrograms_pytorch/train"
    output_dir_val = "spectrograms_pytorch/validation"

    # Process train split
    preprocessor_train = SpectrogramPreprocessor(csv_path, audio_dir, output_dir_train, split='train')
    preprocessor_train.process()

    # Process validation split
    preprocessor_val = SpectrogramPreprocessor(csv_path, audio_dir, output_dir_val, split='validation')
    preprocessor_val.process()
