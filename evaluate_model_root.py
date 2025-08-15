from imports import *
from RootDataset import RootDataset
from ChordCNN import ChordCNN

def validate_model(model_path, csv_path, audio_path, split='train', batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = RootDataset(csv_path=csv_path, dir_path=audio_path, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ChordCNN([1, 32, 64, 128], num_classes=len(ROOT_CLASSES), dropout=0).to(device)
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
    print(classification_report(all_labels, all_preds, target_names=ROOT_CLASSES))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ROOT_CLASSES, yticklabels=ROOT_CLASSES)
    plt.title(f'Matriz de Confusão ({split})')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model_path = "models_root/acc=0.44_loss=1.78_batch=4_lr=0.001_drop=0.1_epochs=250.pt"
    csv_path = "dataset/chords.csv"
    audio_path = "dataset/chords/"
    
    validate_model(model_path, csv_path, audio_path, split='train')
