import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
class FoodDataset(Dataset):
    def __init__(self, data_dir, transform=None, img_size=128):
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size
        self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = []
        self.labels = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        print(f"From {data_dir} loaded {len(self.images)} picture， {len(self.classes)} category")

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
def create_data_loaders_fast(data_root_dir, batch_size=128, img_size=128):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dir = os.path.join(data_root_dir, 'training')
    validation_dir = os.path.join(data_root_dir, 'validation')
    test_dir = os.path.join(data_root_dir, 'test')
    train_dataset = FoodDataset(train_dir, transform=train_transform, img_size=img_size)
    val_dataset = FoodDataset(validation_dir, transform=val_test_transform, img_size=img_size)
    test_dataset = FoodDataset(test_dir, transform=val_test_transform, img_size=img_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    return train_loader, val_loader, test_loader, train_dataset.classes
class UltraFastFoodCNN(nn.Module):
    def __init__(self, num_classes):
        super(UltraFastFoodCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),
            nn.Conv2d(128, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
def train_model_ultra_fast(model, train_loader, val_loader, num_epochs=8):
    from torch.cuda.amp import GradScaler, autocast
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
    scaler = GradScaler()
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.1,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100
    )
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        model.train()
        running_loss = 0.0
        running_corrects = 0
        train_bar = tqdm(train_loader, desc=f'Train', leave=False)
        for batch_idx, (inputs, labels) in enumerate(train_bar):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            if batch_idx % 20 == 0:
                train_bar.set_description(f'Training loss: {loss.item():.3f}')
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.cpu())
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad(), autocast():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc.cpu())
        print(f'Training loss: {loss.item():.3f}|Train Acc: {epoch_acc:.4f} | Verify Acc: {val_epoch_acc:.4f} | LR: {optimizer.param_groups[0]["lr"]:.2e}')
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), 'best_food_model_ultra_fast.pth')
            print(f'Save Best Verification Accuracy: {best_val_acc:.4f}')
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }
def evaluate_fast(model, test_loader):
    model.eval()
    test_corrects = 0
    with torch.no_grad(), torch.cuda.amp.autocast():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
    test_acc = test_corrects.double() / len(test_loader.dataset)
    print(f'Test accuracy: {test_acc:.4f}')
    return test_acc
def plot_training_fast(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], 'b-', label='Training loss')
    plt.plot(history['val_losses'], 'r-', label='Verification loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracies'], 'b-', label='Training accuracy')
    plt.plot(history['val_accuracies'], 'r-', label='Verification accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history_fast.png', dpi=150, bbox_inches='tight')
    plt.close()
def main_ultra_fast():
    data_root_dir = "food_images"
    batch_size = 128
    img_size = 128
    num_epochs = 30

    if not os.path.exists(data_root_dir):
        print(f"Error: Data directory '{data_root_dir}' Does not exist!")
        return
    train_loader, val_loader, test_loader, class_names = create_data_loaders_fast(
        data_root_dir, batch_size, img_size
    )
    print(f"\n Dataset statistics:")
    print(
        f"Category: {len(class_names)} | Train: {len(train_loader.dataset)} | Verify: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    model = UltraFastFoodCNN(num_classes=len(class_names))
    model = model.to(device)
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters()):,}")
    print("Start training...")
    history = train_model_ultra_fast(model, train_loader, val_loader, num_epochs)
    plot_training_fast(history)
    print("Final Test...")
    model.load_state_dict(torch.load('best_food_model_ultra_fast.pth'))
    test_acc = evaluate_fast(model, test_loader)
    print(f"\n Training complete! Accuracy test: {test_acc:.4f}")
def create_pretrained_ultra_fast(num_classes):
    model = models.mobilenet_v3_small(pretrained=True)
    for param in list(model.parameters())[:-15]:
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(model.classifier[0].in_features, num_classes)
    )
    return model
if __name__ == "__main__":
    main_ultra_fast()