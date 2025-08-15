from imports import *
from ChordDataset import ChordDataset
from ChordCNN import ChordCNN

def validate_model(model_path, csv_path, audio_path, split='validation', batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = ChordDataset(csv_path=csv_path, dir_path=audio_path, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ChordCNN([1, 32, 64, 128], num_classes=len(CLASSES), dropout=0).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device).float(), labels.to(device).long()
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Matriz de Confusão ({split})')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.show()

def plot_log(csv_log):
    df = pd.read_csv(csv_log)
    plt.figure(figsize=(10, 8))
    plt.plot(df['epoch'], df['val_accuracy'], marker='o', linestyle='-', color='blue', label='Val Accuracy')
    # plt.plot(df['epoch'], df['val_loss'], marker='x', linestyle='--', color='red', label='Val Loss')
    plt.title('Acurácia e Loss de Validação por Época')
    plt.xlabel('Época')
    plt.ylabel('Acurácia e Loss de Validação')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model_path = "models_chordsOnly/acc=0.58_loss=1.60_batch=32_lr=0.005_drop=0.5_epochs=500.pt"
    csv_path = "dataset/chords.csv"
    audio_path = "dataset/chords/"
    
    validate_model(model_path, csv_path, audio_path, split='train')
    csv_log = "logs_chordsOnly/val_metrics_batch=32_lr=0.005_drop=0.5_epochs=500.csv"
    plot_log(csv_log)
