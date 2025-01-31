#!/usr/bin/env python3

import os
import random
import logging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import segmentation_models_pytorch as smp

# -------------------------------------------------------------------
# 1) LOGGING SETUP
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,  # or logging.INFO if you prefer less verbosity
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# 2) CONFIGURATION
# -------------------------------------------------------------------
CELEBA_IMG_DIR = "D:/PythonProjects/HairStyling-AI/Datasets/CelebAMask-HQ/CelebA-HQ-img/"
CELEBA_MASK_DIR = "D:/PythonProjects/HairStyling-AI/Datasets/CelebAMask-HQ/CelebAMask-HQ-mask-anno/"

NUM_IMAGES = 30000   # e.g., if you have 30,000 images (0..29999)
SUBFOLDER_SIZE = 2000  # Each mask subfolder has 2000 images (0..14 => 15*2000=30000)

TARGET_SIZE = (256, 256)
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

BATCH_SIZE = 4
NUM_EPOCHS = 2  # smaller for debugging; increase as needed
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

# -------------------------------------------------------------------
# 3) HELPER FUNCTIONS
# -------------------------------------------------------------------
def get_image_path(i: int) -> str:
    """Construct path to an image named '0.jpg', '1.jpg', etc."""
    return os.path.join(CELEBA_IMG_DIR, f"{i}.jpg")

def get_hair_mask_path(i: int) -> str:
    """
    If hair masks are split into subfolders named '0', '1', ... '14', 
    each containing 2000 images:
      subfolder = i // 2000
      local_idx = i % 2000
      => filename = e.g. '00000_hair.png', '00001_hair.png', etc.
    """
    subfolder = i // SUBFOLDER_SIZE
    local_idx = i % SUBFOLDER_SIZE
    filename = f"{local_idx:05d}_hair.png"
    return os.path.join(CELEBA_MASK_DIR, str(subfolder), filename)


def compute_iou(preds: torch.Tensor, targets: torch.Tensor, threshold=0.5, eps=1e-7) -> float:
    """
    A custom IoU function for binary segmentation.
    preds:   [B,1,H,W] raw logits
    targets: [B,H,W] or [B,1,H,W] in {0,1}
    """
    probs = torch.sigmoid(preds)
    preds_bin = (probs > threshold).float()

    if targets.dim() == 3:
        targets = targets.unsqueeze(1)

    intersection = (preds_bin * targets).sum(dim=[1,2,3])
    union = preds_bin.sum(dim=[1,2,3]) + targets.sum(dim=[1,2,3]) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

# -------------------------------------------------------------------
# 4) DATASET
# -------------------------------------------------------------------
class HairSegmentationDataset(Dataset):
    """
    Loads (image, mask) pairs for hair segmentation.
    Wraps data loading in try/except + logging for easier debugging.
    """
    def __init__(self, indices, transform=None):
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        img_path = get_image_path(i)
        mask_path = get_hair_mask_path(i)

        logger.debug(f"[__getitem__] idx={idx}, global_id={i}")
        logger.debug(f"  -> Image path: {img_path}")
        logger.debug(f"  -> Mask path:  {mask_path}")

        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error opening image file {img_path}: {e}")
            raise e

        try:
            mask_pil = Image.open(mask_path).convert("L")
        except Exception as e:
            logger.error(f"Error opening mask file {mask_path}: {e}")
            raise e

        if self.transform is not None:
            try:
                img_pil, mask_pil = self.transform(img_pil, mask_pil)
            except Exception as e:
                logger.error(f"Error in transform for idx={idx}, {e}")
                raise e

        # Convert to tensor
        mask_np = np.array(mask_pil, dtype=np.uint8)
        mask_tensor = torch.from_numpy((mask_np > 128).astype(np.uint8)).long()

        img_tensor = T.ToTensor()(img_pil)

        return img_tensor, mask_tensor

# -------------------------------------------------------------------
# 5) TRANSFORM
# -------------------------------------------------------------------
class SegmentationTransform:
    def __init__(self, output_size=(256,256), mode='train'):
        self.output_size = output_size
        self.mode = mode
        self.resize = T.Resize(self.output_size)

    def __call__(self, image_pil, mask_pil):
        image_pil = self.resize(image_pil)
        mask_pil  = self.resize(mask_pil)

        if self.mode == 'train':
            # Example: random horizontal flip
            if random.random() < 0.5:
                image_pil = T.functional.hflip(image_pil)
                mask_pil  = T.functional.hflip(mask_pil)

        return image_pil, mask_pil

# -------------------------------------------------------------------
# 6) TRAIN & VALIDATION
# -------------------------------------------------------------------
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0
    running_iou  = 0.0

    for batch_idx, (images, masks) in enumerate(dataloader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        iou_val = compute_iou(outputs, masks)
        running_iou += iou_val * images.size(0)

        logger.debug(f"[train_one_epoch] batch={batch_idx}, loss={loss.item():.4f}, iou={iou_val:.4f}")

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_iou  = running_iou  / len(dataloader.dataset)
    return epoch_loss, epoch_iou

def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    running_iou  = 0.0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks.float())

            running_loss += loss.item() * images.size(0)
            iou_val = compute_iou(outputs, masks)
            running_iou += iou_val * images.size(0)

            logger.debug(f"[validate_one_epoch] batch={batch_idx}, loss={loss.item():.4f}, iou={iou_val:.4f}")

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_iou  = running_iou  / len(dataloader.dataset)
    return epoch_loss, epoch_iou

# -------------------------------------------------------------------
# 7) MAIN
# -------------------------------------------------------------------
def main():
    logger.info("Starting main...")

    # 7.1) Gather valid indices
    valid_indices = []
    for i in range(NUM_IMAGES):
        img_path  = get_image_path(i)
        mask_path = get_hair_mask_path(i)
        if os.path.exists(img_path) and os.path.exists(mask_path):
            valid_indices.append(i)

    logger.info(f"Found {len(valid_indices)} valid pairs.")
    if not valid_indices:
        logger.error("No valid image/mask pairs found! Exiting.")
        return

    # 7.2) Shuffle & Split
    random.seed(RANDOM_SEED)
    random.shuffle(valid_indices)
    n_total = len(valid_indices)
    train_end = int(TRAIN_RATIO * n_total)
    val_end   = int((TRAIN_RATIO + VAL_RATIO) * n_total)

    train_indices = valid_indices[:train_end]
    val_indices   = valid_indices[train_end:val_end]
    test_indices  = valid_indices[val_end:]

    logger.info(f"Train set: {len(train_indices)}")
    logger.info(f"Val set:   {len(val_indices)}")
    logger.info(f"Test set:  {len(test_indices)}")

    # 7.3) Datasets & DataLoaders
    train_transform = SegmentationTransform(output_size=TARGET_SIZE, mode='train')
    val_transform   = SegmentationTransform(output_size=TARGET_SIZE, mode='val')

    train_dataset = HairSegmentationDataset(train_indices, transform=train_transform)
    val_dataset   = HairSegmentationDataset(val_indices,   transform=val_transform)
    test_dataset  = HairSegmentationDataset(test_indices,  transform=val_transform)

    # For debugging, set num_workers=0 to run in main process -> full traceback if error
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 7.4) Model: DeepLabV3+ with ResNet50
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )

    # 7.5) Loss & optimizer
    loss_fn = smp.losses.DiceLoss(mode='binary')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_iou = -1.0

    logger.info("Starting training loop...")
    # 7.6) Training Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_iou     = validate_one_epoch(model, val_loader, loss_fn, device)

        logger.info(
            f"[Epoch {epoch}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}"
        )

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), "best_hair_seg_model.pth")
            logger.info("  > Model saved (val IoU improved).")

    logger.info("Training completed.")

    # 7.7) Evaluate on test set
    model.load_state_dict(torch.load("best_hair_seg_model.pth", map_location=device))
    test_loss, test_iou = validate_one_epoch(model, test_loader, loss_fn, device)
    logger.info(f"Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}")

    # 7.8) Demo Inference
    if len(test_indices) > 0:
        model.eval()
        sample_i = random.choice(test_indices)
        logger.info(f"Demo inference on index: {sample_i}")

        img_pil = Image.open(get_image_path(sample_i)).convert("RGB")
        mask_pil = Image.open(get_hair_mask_path(sample_i)).convert("L")

        val_tfm = SegmentationTransform(output_size=TARGET_SIZE, mode='val')
        img_resized, mask_resized = val_tfm(img_pil, mask_pil)

        img_tensor = T.ToTensor()(img_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img_tensor)
            pred_prob = torch.sigmoid(pred).cpu().numpy()[0,0]
            pred_mask = (pred_prob > 0.5).astype(np.uint8)

        # Visualization
        plt.figure(figsize=(10,5))
        plt.subplot(1,3,1)
        plt.imshow(img_resized)
        plt.title("Input Image")

        plt.subplot(1,3,2)
        plt.imshow(mask_resized, cmap='gray')
        plt.title("Ground Truth Mask")

        plt.subplot(1,3,3)
        plt.imshow(pred_mask, cmap='gray')
        plt.title("Predicted Mask")

        plt.tight_layout()
        plt.show()

    logger.info("Done.")


# -------------------------------------------------------------------
# ENTRY POINT (important on Windows for multiprocessing)
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
