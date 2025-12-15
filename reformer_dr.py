import os
import time
import math
import random
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from reformer_pytorch import ReformerLM

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

import deepspeed
import kagglehub


# =========================================================
#  CONFIG
# =========================================================

DATASET_SLUG = "ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy"

NUM_CLASSES = 5

# Image / patch
IMAGE_SIZE = 560
PATCH_SIZE = 20   # 560/20 = 28 -> 784 tokens

# Model
EMBED_DIM = 256
DEPTH = 8
NUM_HEADS = 8
MLP_RATIO = 4

# LSH
LSH_BUCKET_SIZE = 14
LSH_N_HASHES = 1

# Chunking
FF_CHUNKS = 1
ATTN_CHUNKS = 1

# CNN stem
STEM_CHANNELS = 64

# -------- Batching (micro vs effective) --------
MICRO_BATCH_SIZE = 16          # per GPU, per step
EFFECTIVE_BATCH_SIZE = 128      # "logical" batch size you want

# Derived:
BATCH_SIZE = MICRO_BATCH_SIZE
GRAD_ACC_STEPS = max(1, EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE)

# Training
NUM_EPOCHS = 30
LEARNING_RATE = 4e-4
MIN_LR = 2e-5
WARMUP_EPOCHS = 3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.0

# Regularization
ATTN_DROPOUT = 0.0
FF_DROPOUT = 0.0

# Data loading
NUM_WORKERS = 4
PIN_MEMORY = True

# DeepSpeed
ZERO_STAGE = 1
USE_FP16 = True

# Seed
SEED = 42


# =========================================================
#  REPRODUCIBILITY
# =========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# =========================================================
#  DATASET DOWNLOAD + PATH RESOLUTION
# =========================================================

def _has_split_dirs(root: str) -> bool:
    return (
        os.path.isdir(os.path.join(root, "train")) and
        os.path.isdir(os.path.join(root, "val"))
    )

def resolve_dataset_paths():
    """
    Downloads the Kaggle dataset via kagglehub and resolves train/val/test folders.
    Progress bars: high-level only + image counting.
    """
    with tqdm(total=4, desc="Dataset setup (Reformer)", unit="step") as pbar:
        # 1) Download (or cache)
        path = kagglehub.dataset_download(DATASET_SLUG)
        pbar.update(1)

        # 2) Detect actual split root
        candidates = [
            path,
            os.path.join(path, "augmented_resized_V2"),
            os.path.join(path, "augmented_resized"),
            os.path.join(path, "resized"),
            os.path.join(path, "train_val_test"),
        ]

        data_root = None
        for c in candidates:
            if _has_split_dirs(c):
                data_root = c
                break

        if data_root is None:
            try:
                for name in os.listdir(path):
                    c = os.path.join(path, name)
                    if os.path.isdir(c) and _has_split_dirs(c):
                        data_root = c
                        break
            except FileNotFoundError:
                data_root = None

        if data_root is None:
            raise FileNotFoundError(
                "Could not find expected train/val folders inside the downloaded dataset. "
                f"Root returned by kagglehub: {path}"
            )

        pbar.update(1)

        train_dir = os.path.join(data_root, "train")
        val_dir   = os.path.join(data_root, "val")
        test_dir  = os.path.join(data_root, "test") if os.path.isdir(os.path.join(data_root, "test")) else None

        # 3) Sanity check class folders
        train_class_dirs = [
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ]
        if len(train_class_dirs) != NUM_CLASSES:
            print(
                f"[WARN] Detected {len(train_class_dirs)} class folders in train. "
                f"Expected {NUM_CLASSES}. Found: {sorted(train_class_dirs)}"
            )

        pbar.update(1)

        # 4) Count images
        def count_images(path_):
            if path_ is None:
                return 0
            exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            total = 0

            class_dirs = [
                os.path.join(path_, d) for d in os.listdir(path_)
                if os.path.isdir(os.path.join(path_, d))
            ]

            for cd in tqdm(class_dirs, desc=f"Counting {os.path.basename(path_)}", leave=False):
                for root, _, files in os.walk(cd):
                    total += sum(f.lower().endswith(exts) for f in files)
            return total

        n_train = count_images(train_dir)
        n_val   = count_images(val_dir)
        n_test  = count_images(test_dir) if test_dir else 0

        pbar.update(1)

    print(f"Dataset cache root: {path}")
    print(f"Resolved data root: {data_root}")
    print(f"Train dir: {train_dir}")
    print(f"Val dir:   {val_dir}")
    if test_dir:
        print(f"Test dir:  {test_dir}")
    print(f"Train images: {n_train}")
    print(f"Val images:   {n_val}")
    if test_dir:
        print(f"Test images:  {n_test}")

    return train_dir, val_dir, test_dir


# =========================================================
#  DATA
# =========================================================

def get_dataloaders(train_dir, val_dir, test_dir=None):
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds   = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_ds  = datasets.ImageFolder(test_dir, transform=val_test_transform) if test_dir else None

    print("Class to idx mapping:", train_ds.class_to_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

    return train_loader, val_loader, test_loader


# =========================================================
#  MODEL COMPONENTS
# =========================================================

class CNNStem(nn.Module):
    """
    Light CNN stem for inductive bias.
    Input: 3 x H x W
    Output: STEM_CHANNELS x H x W
    """
    def __init__(self, in_channels=3, out_channels=STEM_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class PatchEmbed(nn.Module):
    """
    C x H x W -> Conv2d with kernel=stride=patch_size -> (B, N, D)
    """
    def __init__(self, img_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
                 in_channels=STEM_CHANNELS, embed_dim=EMBED_DIM):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)                 # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, N, D)
        return x


class VisionReformerLMStem(nn.Module):
    """
    Vision classifier using ReformerLM's underlying Reformer encoder
    on continuous patch embeddings.
    """
    def __init__(
        self,
        num_classes=NUM_CLASSES,
        img_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        dim=EMBED_DIM,
        depth=DEPTH,
        heads=NUM_HEADS,
        ff_mult=MLP_RATIO,
        bucket_size=LSH_BUCKET_SIZE,
        n_hashes=LSH_N_HASHES,
        attn_dropout=ATTN_DROPOUT,
        ff_dropout=FF_DROPOUT,
        ff_chunks=FF_CHUNKS,
        attn_chunks=ATTN_CHUNKS,
    ):
        super().__init__()

        self.stem = CNNStem(in_channels=3, out_channels=STEM_CHANNELS)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=STEM_CHANNELS,
            embed_dim=dim
        )

        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.pos_drop = nn.Dropout(attn_dropout)

        lm_kwargs = dict(
            num_tokens=1,
            dim=dim,
            depth=depth,
            max_seq_len=num_patches,
            heads=heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            ff_mult=ff_mult,
            lsh_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            post_attn_dropout=attn_dropout,
            ff_chunks=ff_chunks,
            attn_chunks=attn_chunks,
            causal=False,
            use_full_attn=False,
        )

        try:
            self.lm = ReformerLM(**lm_kwargs)
        except TypeError:
            for k in ["use_full_attn", "post_attn_dropout", "causal", "ff_chunks", "attn_chunks"]:
                lm_kwargs.pop(k, None)
            self.lm = ReformerLM(**lm_kwargs)

        encoder = getattr(self.lm, "reformer", None)
        if encoder is None:
            for name in ["model", "net", "core", "encoder"]:
                if hasattr(self.lm, name):
                    encoder = getattr(self.lm, name)
                    break

        if encoder is None:
            raise RuntimeError(
                "Could not locate underlying Reformer module inside ReformerLM. "
                "Check reformer-pytorch version."
            )

        self.encoder = encoder

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.patch_embed(x)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.encoder(x)
        x = x.mean(dim=1)

        x = self.norm(x)
        logits = self.head(x)
        return logits

    def get_config_string(self):
        return (
            f"CNNStem + Reformer encoder "
            f"(depth={DEPTH}, heads={NUM_HEADS}, "
            f"bucket={LSH_BUCKET_SIZE}, hashes={LSH_N_HASHES}, "
            f"ff_chunks={FF_CHUNKS}, attn_chunks={ATTN_CHUNKS})"
        )


# =========================================================
#  LR (manual warmup + cosine) for DeepSpeed optimizer
# =========================================================

def compute_lr(epoch: int) -> float:
    if epoch <= WARMUP_EPOCHS:
        return LEARNING_RATE * epoch / max(1, WARMUP_EPOCHS)

    progress = (epoch - WARMUP_EPOCHS) / max(1, (NUM_EPOCHS - WARMUP_EPOCHS))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + (LEARNING_RATE - MIN_LR) * cosine

def set_lr(ds_optimizer, lr: float):
    for pg in ds_optimizer.param_groups:
        pg["lr"] = lr


# =========================================================
#  TRAIN / EVAL
# =========================================================

def train_one_epoch(model_engine, loader, criterion, device, epoch, num_epochs):
    model_engine.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Train {epoch}/{num_epochs}", leave=False)

    model_dtype = next(model_engine.module.parameters()).dtype

    for images, labels in pbar:
        images = images.to(device, non_blocking=True).to(dtype=model_dtype)
        labels = labels.to(device, non_blocking=True)

        logits = model_engine(images)
        loss = criterion(logits, labels)

        model_engine.backward(loss)
        model_engine.step()

        bs = images.size(0)
        running_loss += loss.item() * bs

        _, preds = logits.max(1)
        total += bs
        correct += preds.eq(labels).sum().item()

        pbar.set_postfix(
            loss=f"{running_loss/max(1,total):.4f}",
            acc=f"{100.0*correct/max(1,total):.2f}%"
        )

    return running_loss / max(1, total), 100.0 * correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=NUM_CLASSES):
    """
    Fixed evaluate:
    - FP16-safe input casting
    - accumulates labels/preds/probs correctly
    - returns epoch_loss, epoch_acc, precision, recall, f1, auc, cm
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels, all_preds, all_probs = [], [], []

    model_dtype = next(model.parameters()).dtype

    for images, labels in loader:
        images = images.to(device, non_blocking=True).to(dtype=model_dtype)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        bs = images.size(0)
        running_loss += loss.item() * bs

        probs = torch.softmax(logits, dim=1)
        _, preds = logits.max(1)

        total += bs
        correct += preds.eq(labels).sum().item()

        all_labels.append(labels.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_probs.append(probs.detach().cpu())

    epoch_loss = running_loss / max(1, total)
    epoch_acc = 100.0 * correct / max(1, total)

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    labels_one_hot = np.eye(NUM_CLASSES)[all_labels]
    try:
        auc = roc_auc_score(labels_one_hot, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    return epoch_loss, epoch_acc, precision, recall, f1, auc, cm


# =========================================================
#  MAIN
# =========================================================

def main():
    train_dir, val_dir, test_dir = resolve_dataset_paths()

    print("\n" + "=" * 60)
    print("  REFORMER VISION CONFIGURATION (DR)")
    print("=" * 60)
    print(f"  Image size: {IMAGE_SIZE}")
    print(f"  Patch size: {PATCH_SIZE} -> {(IMAGE_SIZE // PATCH_SIZE) ** 2} tokens")
    print(f"  Embed dim:  {EMBED_DIM}")
    print(f"  Depth:      {DEPTH}")
    print(f"  Heads:      {NUM_HEADS}")
    print(f"  MLP ratio:  {MLP_RATIO}x")
    print(f"  ATTN_CHUNKS: {ATTN_CHUNKS}")
    print(f"  FF_CHUNKS:   {FF_CHUNKS}")
    print(f"  MICRO_BATCH_SIZE:     {MICRO_BATCH_SIZE}")
    print(f"  EFFECTIVE_BATCH_SIZE: {EFFECTIVE_BATCH_SIZE}")
    print(f"  Derived BATCH_SIZE (micro): {BATCH_SIZE}")
    print(f"  GRAD_ACC_STEPS: {GRAD_ACC_STEPS}")
    print(f"  ZeRO stage: {ZERO_STAGE}")
    print(f"  FP16: {USE_FP16}")
    print(f"  NUM_WORKERS: {NUM_WORKERS}")
    print(f"  PIN_MEMORY:  {PIN_MEMORY}")
    print("=" * 60 + "\n")

    train_loader, val_loader, test_loader = get_dataloaders(train_dir, val_dir, test_dir)

    model = VisionReformerLMStem(num_classes=NUM_CLASSES)
    print("Model:", model.get_config_string())

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")

    ds_config = {
        "train_micro_batch_size_per_gpu": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACC_STEPS,
        "train_batch_size": BATCH_SIZE * GRAD_ACC_STEPS,

        "fp16": {"enabled": USE_FP16},

        "zero_optimization": {
            "stage": ZERO_STAGE,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY
            }
        },

        "gradient_clipping": 1.0,

        # Must be > 0 to avoid modulo-by-zero in DeepSpeed
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    device = model_engine.device
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).to(device)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    val_precs, val_recs, val_f1s, val_aucs = [], [], [], []
    epoch_times = []
    test_cm = None

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        lr = compute_lr(epoch)
        set_lr(optimizer, lr)

        tr_loss, tr_acc = train_one_epoch(
            model_engine, train_loader, criterion, device, epoch, NUM_EPOCHS
        )

        va_loss, va_acc, va_p, va_r, va_f1, va_auc, cm = evaluate(
            model_engine.module, val_loader, criterion, device, num_classes=NUM_CLASSES
        )

        epoch_time = time.time() - start_time

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(va_loss)
        val_accs.append(va_acc)
        val_precs.append(va_p)
        val_recs.append(va_r)
        val_f1s.append(va_f1)
        val_aucs.append(va_auc)
        epoch_times.append(epoch_time)

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"- LR: {lr:.6f} "
            f"- Train loss: {tr_loss:.4f}, acc: {tr_acc:.2f}% "
            f"- Val loss: {va_loss:.4f}, acc: {va_acc:.2f}% "
            f"- Val P/R/F1/AUC: {va_p:.3f}/{va_r:.3f}/{va_f1:.3f}/{va_auc:.3f} "
            f"- Time: {epoch_time:.2f}s"
        )

    if test_loader is not None:
        te_loss, te_acc, te_p, te_r, te_f1, te_auc, test_cm = evaluate(
            model_engine.module, test_loader, criterion, device, num_classes=NUM_CLASSES
        )

        print("\n" + "=" * 60)
        print("  TEST RESULTS (REFORMER)")
        print("=" * 60)
        print(f"  Test loss: {te_loss:.4f}")
        print(f"  Test acc:  {te_acc:.2f}%")
        print(f"  Test P/R/F1: {te_p:.3f}/{te_r:.3f}/{te_f1:.3f}")
        print(f"  Test AUC:  {te_auc:.3f}")
        print("=" * 60)
        print("Confusion matrix:")
        print(test_cm)

    # ---- Plots ----
    epochs = range(1, NUM_EPOCHS + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Val Loss (Reformer Vision + DeepSpeed)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train / Val Accuracy (Reformer Vision + DeepSpeed)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_precs, label="Val Precision (macro)")
    plt.plot(epochs, val_recs, label="Val Recall (macro)")
    plt.plot(epochs, val_f1s, label="Val F1 (macro)")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Val Precision / Recall / F1 (Reformer Vision)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_aucs, label="Val AUC (macro OVR)")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("Validation AUC (Reformer Vision)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    avg_epoch_time = sum(epoch_times) / max(1, len(epoch_times))
    print(f"\nAverage epoch time: {avg_epoch_time:.2f}s")

    plt.figure(figsize=(8, 5))
    plt.plot(list(epochs), epoch_times, marker="o", label="Epoch time (s)")
    plt.axhline(avg_epoch_time, linestyle="--", label=f"Avg = {avg_epoch_time:.2f}s")
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.title("Epoch Time per Epoch (Reformer Vision)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if test_cm is not None:
        plt.figure(figsize=(6, 5))
        plt.imshow(test_cm, interpolation="nearest")
        plt.title("Test Confusion Matrix (Reformer Vision)")
        plt.colorbar()
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
