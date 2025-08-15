from imports import *
from ChordDataset import ChordDataset
from ChordCNN import ChordCNN

CLASSES = ['maj', 'min', 'aug', 'dim', 'sus2', 'sus4', 'dim7', 'maj7', 'min7', 'min7b5', 'seventh', 'sixth']

def model_exists(dir, batch_size, lr, dropout, epochs):
    patterns = [
        re.compile(r"acc=\d+\.\d+_batch=(\d+)_lr=([0-9.]+)_drop=([0-9.]+)_epochs=(\d+)\.pt"),
        re.compile(r"acc=\d+\.\d+_loss=\d+\.\d+_batch=(\d+)_lr=([0-9.]+)_drop=([0-9.]+)_epochs=(\d+)\.pt")
    ]

    for filename in os.listdir(dir):
        for pattern in patterns:
            match = pattern.match(filename)
            if match:
                b, l, d, e = match.groups()
                if int(b) == batch_size and float(l) == lr and float(d) == dropout and int(e) == epochs:
                    return True
    return False


def evaluate_chord(model, dataloader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(device).float(), label.to(device).long()
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

            loss = criterion(outputs, label)
            total_loss += loss.item() * label.size(0)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return accuracy, avg_loss

def train_model():
    csv_path = "dataset/chords.csv"
    audio_path = "dataset/chords/"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs("models_chordsOnly", exist_ok=True)
    os.makedirs("logs_chordsOnly", exist_ok=True)

    EPOCHS = [100, 250, 500]
    BATCH_SIZE = [4, 8, 16, 32, 64, 128, 256, 512]
    LR = [0.5, 0.05, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    DROPOUT = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    for epochs in EPOCHS:
        for batch_size in BATCH_SIZE:
            for lr in LR:
                for dropout in DROPOUT:
                    if model_exists("models_chordsOnly", batch_size, lr, dropout, epochs):
                        print(f"Modelo com batch={batch_size}, lr={lr}, dropout={dropout}, epochs={epochs} já existe. Pulando treino.")
                        continue

                    start = time.time()
                    print(f"\n--- Training: epochs={epochs}, batch_size={batch_size}, lr={lr}, dropout={dropout} ---")

                    train_dataset = ChordDataset(csv_path=csv_path, dir_path=audio_path, split='train')
                    val_dataset = ChordDataset(csv_path=csv_path, dir_path=audio_path, split='validation')

                    if len(train_dataset) == 0 or len(val_dataset) == 0:
                        print("Empty train or val dataset. Skipping.")
                        continue

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                    model = ChordCNN([1, 32, 64, 128], num_classes=len(CLASSES), dropout=dropout).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.CrossEntropyLoss()

                    best_acc = 0.0
                    best_loss = float('inf')
                    val_metrics_per_epoch = []
                    config_suffix = f"batch={batch_size}_lr={lr}_drop={dropout}_epochs={epochs}"
                    temp_model_path = ""

                    for epoch in range(epochs):
                        model.train()
                        for data, label in train_loader:
                            data = data.to(device).float()
                            label = label.to(device).long()

                            optimizer.zero_grad()
                            output = model(data)
                            loss = criterion(output, label)
                            loss.backward()
                            optimizer.step()

                        val_acc, val_loss = evaluate_chord(model, val_loader, device, criterion)
                        val_metrics_per_epoch.append((val_acc, val_loss))
                        print(f"Epoch {epoch} - Val Accuracy: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

                        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
                            old_temp_files = [
                                f for f in os.listdir("models_chordsOnly")
                                if f.endswith(f"temp_{config_suffix}.pt")
                            ]
                            for f in old_temp_files:
                                os.remove(os.path.join("models_chordsOnly", f))

                            best_acc = val_acc
                            best_loss = val_loss
                            temp_model_path = f"models_chordsOnly/acc={best_acc:.2f}_loss={best_loss:.2f}_temp_{config_suffix}.pt"
                            torch.save(model.state_dict(), temp_model_path)
                            print(f"Novo melhor modelo salvo temporariamente: {temp_model_path}")

                    final_model_path = f"models_chordsOnly/acc={best_acc:.2f}_loss={best_loss:.2f}_{config_suffix}.pt"
                    if temp_model_path and os.path.exists(temp_model_path):
                        os.rename(temp_model_path, final_model_path)
                        print(f"Modelo final salvo: {final_model_path}")
                    else:
                        print("Nenhum modelo temporário salvo para renomear.")

                    log_path = f"logs_chordsOnly/val_metrics_{config_suffix}.csv"
                    with open(log_path, 'w') as f:
                        f.write("epoch,val_accuracy,val_loss\n")
                        for i, (acc, loss) in enumerate(val_metrics_per_epoch):
                            f.write(f"{i},{acc:.4f},{loss:.4f}\n")
                    print(f"Métricas por época salvas em: {log_path}")

                    elapsed = time.time() - start
                    hours = int(elapsed // 3600)
                    minutes = int((elapsed % 3600) // 60)
                    seconds = int(elapsed % 60)
                    print(f"Tempo total de treinamento: {hours}h {minutes}m {seconds}s")

if __name__ == "__main__":
    train_model()
