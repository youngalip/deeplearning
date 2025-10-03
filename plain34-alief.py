import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchinfo import summary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import time
import copy

# Configuration
CLASS_NAMES = ["bakso", "gado_gado", "nasi_goreng", "rendang", "soto_ayam"]
NUM_CLASSES = len(CLASS_NAMES)

# Hyperparameters - optimized for limited dataset
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
MAX_SAMPLES = 300  # Limit total samples to 300

print(f"Classes: {CLASS_NAMES}")
print(f"Total classes: {NUM_CLASSES}")

class FoodDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(CLASS_NAMES)}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['filename']
        
        # Try different possible paths
        possible_paths = [
            os.path.join(self.img_dir, img_name),
            os.path.join(self.img_dir, 'images', img_name),
            os.path.join(self.img_dir, 'train', img_name),
            img_name  # If full path is already in filename
        ]
        
        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_name}. Tried paths: {possible_paths}")
        
        image = Image.open(img_path).convert('RGB')
        
        if 'label' in self.df.columns:
            label = self.df.iloc[idx]['label']
            label_idx = self.class_to_idx[label]
        else:
            label_idx = -1  # For test data
        
        if self.transform:
            image = self.transform(image)
        
        if label_idx != -1:
            return image, label_idx
        else:
            return image

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# ==============================
# PLAIN NETWORK (from your code)
# ==============================

class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PlainBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        # NO RESIDUAL CONNECTION - This is the key difference
        out = F.relu(out)
        
        return out

class Plain34(nn.Module):
    def __init__(self, num_classes=5):
        super(Plain34, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage1 = self._make_stage(64, 64, 3, stride=1)
        self.stage2 = self._make_stage(64, 128, 4, stride=2)
        self.stage3 = self._make_stage(128, 256, 6, stride=2)
        self.stage4 = self._make_stage(256, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
        
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        downsample = None
        
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(PlainBlock(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, num_blocks):
            layers.append(PlainBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# ==============================
# RESNET-34 (with Residual Connections)
# ==============================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        # RESIDUAL CONNECTION - This is the key addition
        out += identity
        out = F.relu(out)
        
        return out

class ResNet34(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet34, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage1 = self._make_stage(64, 64, 3, stride=1)
        self.stage2 = self._make_stage(64, 128, 4, stride=2)
        self.stage3 = self._make_stage(128, 256, 6, stride=2)
        self.stage4 = self._make_stage(256, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
        
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        downsample = None
        
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# ==============================
# TRAINING UTILITIES
# ==============================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name):
    """
    Train a model and return training history
    """
    print(f"\n{'='*50}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'='*50}")
    
    model = model.to(device)
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # History tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 20)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation", leave=False)
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_acc = val_running_corrects.double() / len(val_loader.dataset)
        
        # Store history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc.cpu())
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc.cpu())
        
        print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')
        
        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    
    total_time = time.time() - start_time
    print(f'\nTraining complete in {total_time // 60:.0f}m {total_time % 60:.0f}s')
    print(f'Best val Acc: {best_val_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def plot_training_history(plain_history, resnet_history):
    """
    Plot training history comparison between Plain and ResNet models
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(plain_history['train_loss']) + 1)
    
    # Training Loss
    ax1.plot(epochs, plain_history['train_loss'], 'b-', label='Plain-34', linewidth=2)
    ax1.plot(epochs, resnet_history['train_loss'], 'r-', label='ResNet-34', linewidth=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Validation Loss
    ax2.plot(epochs, plain_history['val_loss'], 'b-', label='Plain-34', linewidth=2)
    ax2.plot(epochs, resnet_history['val_loss'], 'r-', label='ResNet-34', linewidth=2)
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Training Accuracy
    ax3.plot(epochs, plain_history['train_acc'], 'b-', label='Plain-34', linewidth=2)
    ax3.plot(epochs, resnet_history['train_acc'], 'r-', label='ResNet-34', linewidth=2)
    ax3.set_title('Training Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True)
    
    # Validation Accuracy
    ax4.plot(epochs, plain_history['val_acc'], 'b-', label='Plain-34', linewidth=2)
    ax4.plot(epochs, resnet_history['val_acc'], 'r-', label='ResNet-34', linewidth=2)
    ax4.set_title('Validation Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, val_loader, device, model_name):
    """
    Evaluate model and return detailed metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Evaluating {model_name}"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Classification report
    print(f"\n{model_name} Classification Report:")
    print("="*50)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()
    
    return all_preds, all_labels

# ==============================
# MAIN EXECUTION
# ==============================

def main():
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("\n" + "="*50)
    print("LOADING DATASET")
    print("="*50)
    
    train_df = pd.read_csv('train.csv')
    print(f"Original training data: {len(train_df)} samples")
    
    # Limit to MAX_SAMPLES for faster training and testing
    if len(train_df) > MAX_SAMPLES:
        # Sample equally from each class if possible
        samples_per_class = MAX_SAMPLES // NUM_CLASSES
        limited_df = pd.DataFrame()
        
        for class_name in CLASS_NAMES:
            class_data = train_df[train_df['label'] == class_name]
            if len(class_data) >= samples_per_class:
                sampled = class_data.sample(n=samples_per_class, random_state=42)
            else:
                sampled = class_data
            limited_df = pd.concat([limited_df, sampled])
        
        # If we still need more samples, randomly sample from remaining
        if len(limited_df) < MAX_SAMPLES:
            remaining_needed = MAX_SAMPLES - len(limited_df)
            remaining_data = train_df[~train_df.index.isin(limited_df.index)]
            if len(remaining_data) > 0:
                additional = remaining_data.sample(n=min(remaining_needed, len(remaining_data)), random_state=42)
                limited_df = pd.concat([limited_df, additional])
        
        train_df = limited_df.reset_index(drop=True)
    
    print(f"Using {len(train_df)} samples for training")
    print("Class distribution:")
    print(train_df['label'].value_counts())
    
    # Visualize class distribution
    plt.figure(figsize=(10, 6))
    train_df['label'].value_counts().plot(kind='bar')
    plt.title('Class Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Create full dataset
    full_dataset = FoodDataset(train_df, img_dir='.', transform=None)  # Temporarily without transform
    
    # Split into train and validation
    train_size = int((1 - VALIDATION_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders with reduced num_workers for Windows compatibility
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # ==============================
    # TAHAP 1: PLAIN-34 TRAINING
    # ==============================
    
    print("\n" + "="*50)
    print("TAHAP 1: TRAINING PLAIN-34")
    print("="*50)
    
    # Create Plain-34 model
    plain_model = Plain34(num_classes=NUM_CLASSES)
    
    # Display model summary
    print("\nPlain-34 Architecture:")
    try:
        summary(plain_model, input_size=(1, 3, IMG_SIZE, IMG_SIZE))
    except:
        pass
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    plain_optimizer = optim.Adam(plain_model.parameters(), lr=LEARNING_RATE)
    
    # Train Plain-34
    plain_model, plain_history = train_model(
        plain_model, train_loader, val_loader, criterion, 
        plain_optimizer, NUM_EPOCHS, device, "Plain-34"
    )
    
    # Save Plain-34 model
    torch.save(plain_model.state_dict(), 'plain34_model.pth')
    
    # ==============================
    # TAHAP 2: RESNET-34 TRAINING
    # ==============================
    
    print("\n" + "="*50)
    print("TAHAP 2: TRAINING RESNET-34")
    print("="*50)
    
    # Create ResNet-34 model
    resnet_model = ResNet34(num_classes=NUM_CLASSES)
    
    # Display model summary
    print("\nResNet-34 Architecture:")
    try:
        summary(resnet_model, input_size=(1, 3, IMG_SIZE, IMG_SIZE))
    except:
        pass
    
    # Setup training components (same hyperparameters for fair comparison)
    resnet_optimizer = optim.Adam(resnet_model.parameters(), lr=LEARNING_RATE)
    
    # Train ResNet-34
    resnet_model, resnet_history = train_model(
        resnet_model, train_loader, val_loader, criterion, 
        resnet_optimizer, NUM_EPOCHS, device, "ResNet-34"
    )
    
    # Save ResNet-34 model
    torch.save(resnet_model.state_dict(), 'resnet34_model.pth')
    
    # ==============================
    # ANALYSIS AND COMPARISON
    # ==============================
    
    print("\n" + "="*50)
    print("ANALYSIS AND COMPARISON")
    print("="*50)
    
    # Plot training history comparison
    plot_training_history(plain_history, resnet_history)
    
    # Evaluate both models
    plain_preds, plain_labels = evaluate_model(plain_model, val_loader, device, "Plain-34")
    resnet_preds, resnet_labels = evaluate_model(resnet_model, val_loader, device, "ResNet-34")
    
    # Final comparison summary
    print("\n" + "="*60)
    print("FINAL COMPARISON SUMMARY")
    print("="*60)
    
    plain_final_acc = plain_history['val_acc'][-1]
    resnet_final_acc = resnet_history['val_acc'][-1]
    
    print(f"Plain-34 Final Validation Accuracy: {plain_final_acc:.4f}")
    print(f"ResNet-34 Final Validation Accuracy: {resnet_final_acc:.4f}")
    print(f"Improvement with Residual Connections: {resnet_final_acc - plain_final_acc:.4f}")
    print(f"Relative Improvement: {((resnet_final_acc - plain_final_acc) / plain_final_acc * 100):.2f}%")
    
    # Best accuracies
    plain_best_acc = max(plain_history['val_acc'])
    resnet_best_acc = max(resnet_history['val_acc'])
    
    print(f"\nBest Validation Accuracies:")
    print(f"Plain-34: {plain_best_acc:.4f}")
    print(f"ResNet-34: {resnet_best_acc:.4f}")
    
    # Training efficiency analysis
    print(f"\nTraining Loss Convergence:")
    print(f"Plain-34 Final Training Loss: {plain_history['train_loss'][-1]:.4f}")
    print(f"ResNet-34 Final Training Loss: {resnet_history['train_loss'][-1]:.4f}")

if __name__ == "__main__":

    main()
