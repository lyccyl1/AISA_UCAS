import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as Transforms
from tqdm import tqdm
from utils import Xception  # 保留你原来的模型定义

class SingleFolderDataset(Dataset):
    """从一个文件夹读取所有图片，并根据文件名决定标签"""
    def __init__(self, root_dir, transform=None):
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        fname = os.path.basename(path)
        label = 0 if ('Celeb-real' in fname or 'YouTube-real' in fname) else 1
        return img, label


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder",   type=str,   default="/data0/user/ycliu/AISA_data/Celeb-DF-v2/target_pictures/results",
                        help="图片所在的文件夹路径，所有图片在同一目录下")
    parser.add_argument("--save-path",     type=str,   default="./checkpoints",
                        help="训练完成后模型权重保存目录")
    parser.add_argument("--batch-size",    type=int,   default=16,
                        help="训练和验证时的 batch size")
    parser.add_argument("--epochs",        type=int,   default=20,
                        help="最大训练轮数")
    parser.add_argument("--lr",            type=float, default=1e-4,
                        help="初始学习率")
    parser.add_argument("--weight-decay",  type=float, default=1e-4,
                        help="优化器权重衰减")
    parser.add_argument("--step-size",     type=int,   default=10,
                        help="lr_scheduler 每隔多少 epoch 衰减一次")
    parser.add_argument("--gamma",         type=float, default=0.5,
                        help="lr 衰减倍数")
    parser.add_argument("--val-split",     type=float, default=0.1,
                        help="验证集占总数据的比例（0~1）")
    parser.add_argument("--pretrained-model", type=str, default="/data0/user/ycliu/AISA_UCAS/detection/utils/weights.ckpt",
                        help="预训练模型路径，如果提供，则加载后继续训练")
    return parser.parse_args()


def get_transforms():
    return Transforms.Compose([
        Transforms.Resize((299, 299)),
        Transforms.ToTensor(),
        Transforms.Normalize([0.5]*3, [0.5]*3),
    ])


def get_dataloaders(opts):
    tfm = get_transforms()
    full_ds = SingleFolderDataset(opts.data_folder, tfm)
    val_size = int(len(full_ds) * opts.val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    train_loader = DataLoader(
        train_ds,
        batch_size=opts.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=opts.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model():
    model = Xception()
    model.fc = nn.Linear(2048, 1)  # 二分类
    return model


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    correct = 0
    total = len(loader.dataset)
    for imgs, labels in tqdm(loader, desc="Validation", leave=False):
        imgs = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def train():
    opts = get_opts()
    os.makedirs(opts.save_path, exist_ok=True)

    train_loader, val_loader = get_dataloaders(opts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)

    # 加载预训练模型（如果提供）
    if opts.pretrained_model:
        assert os.path.isfile(opts.pretrained_model), f"预训练模型文件不存在: {opts.pretrained_model}"
        checkpoint = torch.load(opts.pretrained_model, map_location=device)
        model.load_state_dict(checkpoint)
        print(f"已加载预训练模型: {opts.pretrained_model}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=opts.lr,
        weight_decay=opts.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=opts.step_size,
        gamma=opts.gamma
    )

    best_val_acc = 0.0
    for epoch in range(1, opts.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{opts.epochs} - Training"):  
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}/{opts.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(opts.save_path, "best_model.pth"))
        scheduler.step()

    torch.save(model.state_dict(), os.path.join(opts.save_path, "last_model.pth"))

if __name__ == "__main__":
    train()
