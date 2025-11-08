# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import ImprovedCNN, ResNetMedical
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions

# --- Model Selection ---
USE_EFFICIENTNET = False  # True = EfficientNet-B0, False = ResNet18

# --- Hyperparameters ---
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
CLIP_NORM = 1.0

# Scheduler parameters
T_0 = 10
T_MULT = 2
ETA_MIN = 1e-6

#Menampilkan plot riwayat training dan validasi setelah training selesai.

def train():
    # 1. Memuat Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Inisialisasi Model (Pretrained)
    if USE_EFFICIENTNET:
        model = ImprovedCNN(in_channels=in_channels, num_classes=num_classes, use_pretrained=True)
        model_name = "EfficientNet-B0"
    else:
        model = ResNetMedical(in_channels=in_channels, num_classes=num_classes, use_pretrained=True)
        model_name = "ResNet18"
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*70}")
    print(f"Model: {model_name} (Pretrained on ImageNet)")
    print(f"Total Parameters: {total_params:,}")
    print(f"{'='*70}")
    
    # 3. Hitung pos_weight untuk menangani class imbalance
    class_counts = [0, 0]
    for _, labels in train_loader:
        for label in labels:
            class_counts[int(label.item())] += 1
    
    pos_weight_value = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_value])
    
    print(f"\n--- Class Distribution ---")
    print(f"Class 0 (Cardiomegaly): {class_counts[0]} samples")
    print(f"Class 1 (Pneumothorax): {class_counts[1]} samples")
    print(f"Pos Weight: {pos_weight_value:.4f}")
    
    # 4. Loss Function dengan pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 5. Optimizer SGD
    optimizer = optim.SGD(
        model.parameters(), 
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )
    
    # 6. Learning Rate Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_MULT,
        eta_min=ETA_MIN
    )
    
    print(f"\n--- Training Configuration ---")
    print(f"Model: {model_name} (Pretrained)")
    print(f"Optimizer: SGD with Nesterov Momentum")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Momentum: {MOMENTUM}")
    print(f"Weight Decay: {WEIGHT_DECAY}")
    print(f"Gradient Clipping: {CLIP_NORM}")
    print(f"Scheduler: CosineAnnealingWarmRestarts (T_0={T_0}, T_mult={T_MULT})")
    print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")
    
    # Inisialisasi list untuk menyimpan history
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    learning_rates_history = []
    
    print("\n--- Memulai Training ---")
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            labels = labels.float()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # Hitung training accuracy
            predicted = (outputs > 0).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Update learning rate scheduler
        scheduler.step()
        
        # Simpan learning rate saat ini
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates_history.append(current_lr)
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # --- Fase Validasi ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images
                labels = labels.float()
                
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
                
                predicted = (outputs > 0).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Simpan history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
              f"LR: {current_lr:.6f}")

    print("--- Training Selesai ---")
    
    # Plot learning rate schedule
    plt.figure(figsize=(10, 4))
    plt.plot(learning_rates_history)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule (CosineAnnealingWarmRestarts)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Tampilkan plot
    plot_training_history(train_losses_history, val_losses_history, 
                         train_accs_history, val_accs_history)

    # Visualisasi prediksi pada 10 gambar random dari validation set
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)

if _name_ == '_main_':
    train()