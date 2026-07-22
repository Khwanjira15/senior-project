"""Lightweight inference service for the chest X-ray demo web app."""

from __future__ import annotations

import base64
import hashlib
import os
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

try:
    import pillow_heif
except ImportError:  # pragma: no cover - optional dependency for deploy targets
    pillow_heif = None

try:
    import pydicom
except ImportError:  # pragma: no cover - optional dependency for deploy targets
    pydicom = None

if pillow_heif is not None:
    pillow_heif.register_heif_opener()


BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive"
RESULTS_BASELINE_DIR = BASE_DIR / "Results_PyTorch_Baseline4"
RESULTS_ATTENTION_DIR = BASE_DIR / "Results_PyTorch_Attention4"
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CLASS_NAMES = ["normal", "pneumonia", "tuberculosis"]
ATTENTION_TYPES = ["Baseline", "SE", "CBAM", "ECA", "BAM", "CoordAtt"]
MODEL_OPTIONS = [
    {"key": "ResNet50", "label": "ResNet50"},
    {"key": "DenseNet121", "label": "DenseNet121"},
    {"key": "EfficientNet-B0", "label": "EfficientNet-B0"},
    {"key": "MobileNet", "label": "MobileNetV2"},
    {"key": "MobileNetV3-Large", "label": "MobileNetV3-Large"},
    {"key": "EfficientNetV2-S", "label": "EfficientNetV2-S"},
]
MODEL_LABELS = {item["key"]: item["label"] for item in MODEL_OPTIONS}

MODEL_CACHE: Dict[Tuple[str, str], Tuple[nn.Module, str]] = {}
CHECKPOINT_DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "clinical-ai-checkpoints"

THORAX_CROP_BOX = (0.15, 0.05, 0.85, 0.95)
CAM_LUNG_PRIOR_STRENGTH = 0.60
LUNG_MASK_MARGIN = 0.10
LUNG_CONTEXT_BLEND = 0.08
MASK_THRESHOLD = 32
SCORECAM_MAX_MAPS = 8
SUPPORTED_UPLOAD_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jfif",
    ".bmp",
    ".dib",
    ".tif",
    ".tiff",
    ".webp",
    ".heic",
    ".heif",
    ".avif",
    ".gif",
    ".pdf",
    ".dcm",
    ".dicom",
    ".ima",
}
DIRECT_PIL_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".jpe",
    ".jfif",
    ".bmp",
    ".dib",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
}
SIPS_CONVERTIBLE_EXTENSIONS = {".heic", ".heif", ".avif", ".pdf"}
QUICKLOOK_CONVERTIBLE_EXTENSIONS = {".dcm", ".dicom", ".ima"}


def discover_class_names() -> list[str]:
    train_dir = ARCHIVE_DIR / "train"
    if train_dir.is_dir():
        discovered = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
        if discovered:
            return discovered
    return DEFAULT_CLASS_NAMES


CLASS_NAMES = discover_class_names()

eval_transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def find_lung_masks_dir() -> Optional[Path]:
    candidates = [
        BASE_DIR / "lung_masks",
        BASE_DIR / "archive_masks",
        BASE_DIR / "masks",
        BASE_DIR / "lung_segmentation_masks",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return None


LUNG_MASKS_DIR = find_lung_masks_dir()


class ThoraxCrop:
    def __init__(self, crop_box=THORAX_CROP_BOX):
        self.left, self.top, self.right, self.bottom = crop_box

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        left = int(width * self.left)
        top = int(height * self.top)
        right = int(width * self.right)
        bottom = int(height * self.bottom)
        return img.crop((left, top, right, bottom))


thorax_crop = ThoraxCrop()


def resolve_mask_path(img_path: Path) -> Optional[Path]:
    if LUNG_MASKS_DIR is None:
        return None

    rel_path = img_path.relative_to(ARCHIVE_DIR)
    rel_no_ext = rel_path.with_suffix("")

    for suffix in ["", "_mask", "-mask"]:
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            candidate = LUNG_MASKS_DIR / f"{rel_no_ext}{suffix}{ext}"
            if candidate.exists():
                return candidate
    return None


def prepare_focused_image(img: Image.Image, mask_path: Optional[Path] = None):
    if mask_path is None:
        return thorax_crop(img), None

    mask_img = Image.open(mask_path).convert("L").resize(img.size, Image.Resampling.NEAREST)
    mask_arr = np.array(mask_img, dtype=np.uint8)
    binary_mask = mask_arr > MASK_THRESHOLD

    if not binary_mask.any():
        return thorax_crop(img), None

    ys, xs = np.where(binary_mask)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    margin_x = max(4, int(width * LUNG_MASK_MARGIN))
    margin_y = max(4, int(height * LUNG_MASK_MARGIN))

    x_min = max(0, x_min - margin_x)
    x_max = min(img.size[0] - 1, x_max + margin_x)
    y_min = max(0, y_min - margin_y)
    y_max = min(img.size[1] - 1, y_max + margin_y)

    img_crop = np.array(img.crop((x_min, y_min, x_max + 1, y_max + 1)), dtype=np.float32)
    mask_crop = binary_mask[y_min : y_max + 1, x_min : x_max + 1].astype(np.float32)

    alpha = LUNG_CONTEXT_BLEND + (1.0 - LUNG_CONTEXT_BLEND) * mask_crop[..., None]
    focused = np.clip(img_crop * alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(focused), mask_crop


def get_focused_image_and_mask(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    try:
        mask_path = resolve_mask_path(img_path)
    except ValueError:
        mask_path = None
    return prepare_focused_image(img, mask_path)


def load_raw_image(img_path: Path):
    img, _ = get_focused_image_and_mask(img_path)
    img = img.resize(IMAGE_SIZE)
    return np.array(img)


def normalize_cam(cam):
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    return cam


def build_lung_prior_mask():
    yy, xx = np.mgrid[0 : IMAGE_SIZE[0], 0 : IMAGE_SIZE[1]].astype(np.float32)
    yy /= max(IMAGE_SIZE[0] - 1, 1)
    xx /= max(IMAGE_SIZE[1] - 1, 1)
    left = np.exp(-(((xx - 0.34) ** 2) / 0.018 + ((yy - 0.48) ** 2) / 0.06))
    right = np.exp(-(((xx - 0.66) ** 2) / 0.018 + ((yy - 0.48) ** 2) / 0.06))
    prior = left + right
    prior = cv2.GaussianBlur(prior, (0, 0), sigmaX=7, sigmaY=7)
    return prior / (prior.max() + 1e-8)


LUNG_PRIOR_MASK = build_lung_prior_mask()


def overlay_cam(raw_img, heatmap):
    heatmap = cv2.resize(heatmap, IMAGE_SIZE)
    heatmap = np.uint8(255 * normalize_cam(heatmap))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    raw_bgr = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(raw_bgr, 0.6, heatmap, 0.4, 0)


def get_cam_guidance_mask(img_path: Path):
    _, mask = get_focused_image_and_mask(img_path)
    if mask is None:
        return LUNG_PRIOR_MASK

    resized_mask = cv2.resize(mask.astype(np.float32), IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.GaussianBlur(resized_mask, (0, 0), sigmaX=3, sigmaY=3)
    return resized_mask / (resized_mask.max() + 1e-8)


def apply_lung_prior_to_cam(cam, img_path: Optional[Path] = None, strength=CAM_LUNG_PRIOR_STRENGTH):
    if strength <= 0:
        return normalize_cam(cv2.resize(cam, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR))

    prior = get_cam_guidance_mask(img_path) if img_path is not None else LUNG_PRIOR_MASK
    cam = normalize_cam(cam)
    cam = cv2.resize(cam, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    prior = cv2.resize(prior, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    guided = cam * ((1.0 - strength) + strength * prior)
    guided = np.where(guided >= 0.08, guided, 0.0)
    return normalize_cam(guided)


def disable_inplace_relu(model):
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=False),
            nn.Linear(hidden, channels),
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = torch.mean(x, dim=(2, 3))
        maxv = torch.amax(x, dim=(2, 3))
        ca = self.mlp(avg) + self.mlp(maxv)
        ca = torch.sigmoid(ca).view(b, c, 1, 1)
        x = x * ca
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv = torch.amax(x, dim=1, keepdim=True)
        sa = torch.cat([avg, maxv], dim=1)
        sa = torch.sigmoid(self.conv(sa))
        return x * sa


class ECA(nn.Module):
    def __init__(self, _channels, k_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        y = torch.mean(x, dim=(2, 3), keepdim=True)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = torch.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y


class BAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=False),
            nn.Linear(channels // 4, channels),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        attn = torch.mean(x, dim=(2, 3))
        attn = torch.sigmoid(self.fc(attn)).view(b, c, 1, 1)
        return x * attn


class CoordAtt(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        return x * torch.sigmoid(self.conv1(x))


def get_attention(attn_type, channels):
    if attn_type == "SE":
        return SEBlock(channels)
    if attn_type == "CBAM":
        return CBAM(channels)
    if attn_type == "ECA":
        return ECA(channels)
    if attn_type == "BAM":
        return BAM(channels)
    if attn_type == "CoordAtt":
        return CoordAtt(channels)
    return nn.Identity()


class ResNet50Attention(nn.Module):
    def __init__(self, num_classes, attention_type):
        super().__init__()
        base = models.resnet50(weights=None)
        disable_inplace_relu(base)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.attn = get_attention(attention_type, 2048)
        self.avgpool = base.avgpool
        self.fc = nn.Sequential(
            nn.Linear(base.fc.in_features, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class DenseNet121Attention(nn.Module):
    def __init__(self, num_classes, attention_type):
        super().__init__()
        base = models.densenet121(weights=None)
        disable_inplace_relu(base)
        self.features = base.features
        self.attn = get_attention(attention_type, 1024)
        self.classifier = nn.Sequential(
            nn.Linear(base.classifier.in_features, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=False)
        x = self.attn(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)


class EfficientNetB0Attention(nn.Module):
    def __init__(self, num_classes, attention_type):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        disable_inplace_relu(base)
        self.features = base.features
        self.attn = get_attention(attention_type, 1280)
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(base.classifier[1].in_features, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class MobileNetV2Attention(nn.Module):
    def __init__(self, num_classes, attention_type):
        super().__init__()
        base = models.mobilenet_v2(weights=None)
        disable_inplace_relu(base)
        self.features = base.features
        self.attn = get_attention(attention_type, 1280)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        in_features = base.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class MobileNetV3LargeAttention(nn.Module):
    def __init__(self, num_classes, attention_type):
        super().__init__()
        base = models.mobilenet_v3_large(weights=None)
        disable_inplace_relu(base)
        self.features = base.features
        self.attn = get_attention(attention_type, 960)
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(960, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class EfficientNetV2SAttention(nn.Module):
    def __init__(self, num_classes, attention_type):
        super().__init__()
        base = models.efficientnet_v2_s(weights=None)
        disable_inplace_relu(base)
        self.features = base.features
        self.attn = get_attention(attention_type, 1280)
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base.classifier[1].in_features, 256),
            nn.SiLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(model_name: str, attention_type: str):
    use_attention = attention_type != "Baseline"

    if model_name == "ResNet50":
        if use_attention:
            return ResNet50Attention(len(CLASS_NAMES), attention_type).to(DEVICE), "attn"
        model = models.resnet50(weights=None)
        disable_inplace_relu(model)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(256, len(CLASS_NAMES)),
        )
        return model.to(DEVICE), "layer4.2.conv3"

    if model_name == "DenseNet121":
        if use_attention:
            return DenseNet121Attention(len(CLASS_NAMES), attention_type).to(DEVICE), "attn"
        model = models.densenet121(weights=None)
        disable_inplace_relu(model)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier.in_features, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(256, len(CLASS_NAMES)),
        )
        return model.to(DEVICE), "features.norm5"

    if model_name == "EfficientNet-B0":
        if use_attention:
            return EfficientNetB0Attention(len(CLASS_NAMES), attention_type).to(DEVICE), "attn"
        model = models.efficientnet_b0(weights=None)
        disable_inplace_relu(model)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(256, len(CLASS_NAMES)),
        )
        return model.to(DEVICE), "features.8"

    if model_name == "MobileNet":
        if use_attention:
            return MobileNetV2Attention(len(CLASS_NAMES), attention_type).to(DEVICE), "attn"
        model = models.mobilenet_v2(weights=None)
        disable_inplace_relu(model)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(256, len(CLASS_NAMES)),
        )
        return model.to(DEVICE), "features.18"

    if model_name == "MobileNetV3-Large":
        if use_attention:
            return MobileNetV3LargeAttention(len(CLASS_NAMES), attention_type).to(DEVICE), "attn"
        model = models.mobilenet_v3_large(weights=None)
        disable_inplace_relu(model)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, len(CLASS_NAMES)),
        )
        return model.to(DEVICE), "features.16"

    if model_name == "EfficientNetV2-S":
        if use_attention:
            return EfficientNetV2SAttention(len(CLASS_NAMES), attention_type).to(DEVICE), "attn"
        model = models.efficientnet_v2_s(weights=None)
        disable_inplace_relu(model)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.SiLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(256, len(CLASS_NAMES)),
        )
        return model.to(DEVICE), "features.7"

    raise ValueError(f"Unsupported model: {model_name}")


def get_checkpoint_path(model_name: str, attention_type: str) -> Path:
    if attention_type == "Baseline":
        return RESULTS_BASELINE_DIR / f"{model_name}_Baseline" / f"{model_name}_final.pth"
    return RESULTS_ATTENTION_DIR / f"{model_name}_{attention_type}" / f"{model_name}_final.pth"


def get_checkpoint_env_var_name(model_name: str, attention_type: str) -> str:
    normalized_model = model_name.upper().replace("-", "_")
    normalized_attention = attention_type.upper().replace("-", "_")
    return f"CHECKPOINT_URL_{normalized_model}_{normalized_attention}"


def get_external_checkpoint_url(model_name: str, attention_type: str) -> Optional[str]:
    env_var_name = get_checkpoint_env_var_name(model_name, attention_type)
    return os.getenv(env_var_name)


def has_checkpoint_source(model_name: str, attention_type: str) -> bool:
    checkpoint_path = get_checkpoint_path(model_name, attention_type)
    if checkpoint_path.exists():
        return True
    return bool(get_external_checkpoint_url(model_name, attention_type))


def download_external_checkpoint(model_name: str, attention_type: str, checkpoint_url: str) -> Path:
    CHECKPOINT_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(checkpoint_url.encode("utf-8")).hexdigest()[:16]
    checkpoint_path = CHECKPOINT_DOWNLOAD_DIR / f"{model_name}_{attention_type}_{cache_key}.pth"
    if checkpoint_path.exists():
        return checkpoint_path

    with urllib.request.urlopen(checkpoint_url) as response, checkpoint_path.open("wb") as output_file:
        output_file.write(response.read())
    return checkpoint_path


def resolve_checkpoint_path(model_name: str, attention_type: str) -> Path:
    checkpoint_path = get_checkpoint_path(model_name, attention_type)
    if checkpoint_path.exists():
        return checkpoint_path

    checkpoint_url = get_external_checkpoint_url(model_name, attention_type)
    if checkpoint_url:
        return download_external_checkpoint(model_name, attention_type, checkpoint_url)

    env_var_name = get_checkpoint_env_var_name(model_name, attention_type)
    raise FileNotFoundError(
        "Model checkpoint not found in the deployed app. Add the local file to the repository or set the "
        f"environment variable {env_var_name} to a downloadable .pth URL. Missing local file: {checkpoint_path}"
    )


def get_model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def get_available_model_configs():
    configs = []
    for model in MODEL_OPTIONS:
        for attention_type in ATTENTION_TYPES:
            checkpoint_path = get_checkpoint_path(model["key"], attention_type)
            if has_checkpoint_source(model["key"], attention_type):
                configs.append(
                    {
                        "model_name": model["key"],
                        "model_label": model["label"],
                        "attention_type": attention_type,
                        "checkpoint_path": checkpoint_path,
                    }
                )
    return configs


def get_recommended_model_config():
    preferred_order = [
        ("MobileNet", "ECA"),
        ("MobileNet", "Baseline"),
        ("MobileNetV3-Large", "ECA"),
        ("MobileNetV3-Large", "Baseline"),
        ("EfficientNet-B0", "ECA"),
        ("EfficientNet-B0", "Baseline"),
    ]
    available_map = {
        (config["model_name"], config["attention_type"]): config
        for config in get_available_model_configs()
    }
    for key in preferred_order:
        if key in available_map:
            return available_map[key]
    return next(iter(available_map.values()), None)


def get_target_module(model, module_name):
    return dict(model.named_modules())[module_name]


class CAMExtractor:
    def __init__(self, model, target_module_name):
        self.target_module = get_target_module(model, target_module_name)
        self.activations = None
        self.gradients = None
        self.tensor_grad_handle = None
        self.fwd_handle = self.target_module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inp, out):
        tensor_out = out[0] if isinstance(out, (tuple, list)) else out
        self.activations = tensor_out.detach().clone()
        if self.tensor_grad_handle is not None:
            self.tensor_grad_handle.remove()
            self.tensor_grad_handle = None
        if tensor_out.requires_grad:
            self.tensor_grad_handle = tensor_out.register_hook(self._tensor_backward_hook)

    def _tensor_backward_hook(self, grad):
        self.gradients = grad.detach().clone()

    def remove(self):
        self.fwd_handle.remove()
        if self.tensor_grad_handle is not None:
            self.tensor_grad_handle.remove()
            self.tensor_grad_handle = None


def get_logits_only(outputs):
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def get_gradcam(model, input_tensor, last_conv, img_path=None):
    extractor = CAMExtractor(model, last_conv)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = get_logits_only(model(input_tensor))
        class_idx = torch.argmax(logits, dim=1)
        logits[0, class_idx].backward()
        if extractor.gradients is None:
            raise RuntimeError("GradCAM gradients were not captured.")
        activations = extractor.activations[0]
        gradients = extractor.gradients[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0).detach().cpu().numpy()
        return apply_lung_prior_to_cam(cam, img_path=img_path)
    finally:
        extractor.remove()


def get_gradcam_plus(model, input_tensor, last_conv, img_path=None):
    extractor = CAMExtractor(model, last_conv)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = get_logits_only(model(input_tensor))
        class_idx = torch.argmax(logits, dim=1)
        logits[0, class_idx].backward(retain_graph=True)
        if extractor.gradients is None:
            raise RuntimeError("GradCAM++ gradients were not captured.")
        activations = extractor.activations[0]
        gradients = extractor.gradients[0]
        grads_power_2 = gradients**2
        grads_power_3 = gradients**3
        sum_activations = activations.sum(dim=(1, 2), keepdim=True)
        alphas = grads_power_2 / (2.0 * grads_power_2 + sum_activations * grads_power_3 + 1e-8)
        weights = (alphas * F.relu(gradients)).sum(dim=(1, 2), keepdim=True)
        cam = (weights * activations).sum(dim=0).detach().cpu().numpy()
        return apply_lung_prior_to_cam(cam, img_path=img_path)
    finally:
        extractor.remove()


def get_scorecam(model, input_tensor, last_conv, img_path=None):
    extractor = CAMExtractor(model, last_conv)
    model.eval()
    with torch.no_grad():
        logits = get_logits_only(model(input_tensor))
        class_idx = torch.argmax(logits, dim=1).item()
        activations = extractor.activations[0]
    extractor.remove()

    act = activations.detach().cpu().numpy()
    cams = []
    scores = []
    for i in range(min(act.shape[0], SCORECAM_MAX_MAPS)):
        fmap = normalize_cam(cv2.resize(act[i], IMAGE_SIZE))
        mask = torch.tensor(fmap, dtype=input_tensor.dtype, device=DEVICE).unsqueeze(0).unsqueeze(0)
        masked_input = input_tensor * mask
        with torch.no_grad():
            masked_logits = get_logits_only(model(masked_input))
            score = F.softmax(masked_logits, dim=1)[0, class_idx].item()
        cams.append(fmap)
        scores.append(score)

    cam = np.zeros(IMAGE_SIZE, dtype=np.float32)
    for fmap, score in zip(cams, scores):
        cam += score * fmap
    return apply_lung_prior_to_cam(cam, img_path=img_path)


def image_to_data_url(image_bgr):
    ok, buffer = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image.")
    data = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def convert_with_sips(upload_path: Path) -> Path:
    output_path = upload_path.with_suffix(".png")
    result = subprocess.run(
        ["sips", "-s", "format", "png", str(upload_path), "--out", str(output_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "sips conversion failed")
    return output_path


def convert_dicom_with_pydicom(upload_path: Path) -> Path:
    if pydicom is None:
        raise RuntimeError("pydicom is not installed.")

    dataset = pydicom.dcmread(str(upload_path))
    pixels = dataset.pixel_array.astype(np.float32)

    if pixels.ndim == 3:
        pixels = pixels[0]

    pixels -= pixels.min()
    pixels /= max(pixels.max(), 1e-8)
    pixels = (pixels * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(pixels).convert("RGB")

    output_path = upload_path.with_suffix(".png")
    image.save(output_path)
    return output_path


def convert_with_quicklook(upload_path: Path) -> Path:
    preview_dir = upload_path.parent / "quicklook-preview"
    preview_dir.mkdir(exist_ok=True)
    result = subprocess.run(
        ["qlmanage", "-t", "-s", "1200", "-o", str(preview_dir), str(upload_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Quick Look conversion failed")

    candidates = sorted(preview_dir.glob(f"{upload_path.name}*.png"))
    if not candidates:
        candidates = sorted(preview_dir.glob("*.png"))
    if not candidates:
        raise RuntimeError("Quick Look did not generate a preview image.")
    return candidates[0]


@dataclass
class PredictionResult:
    filename: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    preview_image: str
    gradcam: str
    gradcampp: str
    scorecam: str


def load_model(model_name: str, attention_type: str):
    cache_key = (model_name, attention_type)
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    checkpoint_path = resolve_checkpoint_path(model_name, attention_type)

    model, last_conv = build_model(model_name, attention_type)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    MODEL_CACHE[cache_key] = (model, last_conv)
    return model, last_conv


def ensure_supported_image(upload_path: Path) -> Path:
    suffix = upload_path.suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise RuntimeError(f"Unsupported file type: {suffix or 'no extension'}. Supported extensions: {supported}")

    if suffix in DIRECT_PIL_EXTENSIONS:
        try:
            with Image.open(upload_path) as img:
                img.verify()
            return upload_path
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise RuntimeError(f"Uploaded image could not be opened: {exc}") from exc

    if suffix in SIPS_CONVERTIBLE_EXTENSIONS:
        try:
            return convert_with_sips(upload_path)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise RuntimeError(f"File conversion failed for {suffix}: {exc}") from exc

    if suffix in QUICKLOOK_CONVERTIBLE_EXTENSIONS:
        if pydicom is not None:
            try:
                return convert_dicom_with_pydicom(upload_path)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                if os.name != "posix" or "darwin" not in os.sys.platform:
                    raise RuntimeError(f"DICOM conversion failed with pydicom: {exc}") from exc

        if "darwin" not in os.sys.platform:
            raise RuntimeError(
                "DICOM upload on this deployment requires the optional Python package 'pydicom'. "
                "The macOS Quick Look fallback is not available on Linux."
            )

        try:
            return convert_with_quicklook(upload_path)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise RuntimeError(
                "DICOM conversion failed. This machine does not have a native DICOM decoder in the Python "
                f"environment, and Quick Look could not render the file: {exc}"
            ) from exc

    raise RuntimeError(f"File type {suffix} is recognized but not configured for inference yet.")


def save_upload_to_temp(file_storage) -> Path:
    suffix = Path(file_storage.filename or "upload.png").suffix or ".png"
    temp_dir = Path(tempfile.mkdtemp(prefix="senior-project-upload-"))
    upload_path = temp_dir / f"input{suffix.lower()}"
    file_storage.save(upload_path)
    return ensure_supported_image(upload_path)


def save_named_bytes_to_temp(filename: str, data: bytes) -> Path:
    suffix = Path(filename or "upload.png").suffix or ".png"
    temp_dir = Path(tempfile.mkdtemp(prefix="senior-project-upload-"))
    upload_path = temp_dir / f"input{suffix.lower()}"
    upload_path.write_bytes(data)
    return ensure_supported_image(upload_path)


def predict_image(image_path: Path, model_name: str, attention_type: str) -> PredictionResult:
    model, last_conv = load_model(model_name, attention_type)
    pil_img, _ = get_focused_image_and_mask(image_path)
    input_tensor = eval_transform(pil_img).unsqueeze(0).to(DEVICE)
    raw_img = np.array(pil_img.resize(IMAGE_SIZE))

    with torch.no_grad():
        logits = get_logits_only(model(input_tensor))
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    gradcam = overlay_cam(raw_img, get_gradcam(model, input_tensor, last_conv, img_path=image_path))
    gradcampp = overlay_cam(raw_img, get_gradcam_plus(model, input_tensor, last_conv, img_path=image_path))
    scorecam = overlay_cam(raw_img, get_scorecam(model, input_tensor, last_conv, img_path=image_path))

    preview_rgb = cv2.cvtColor(cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2RGB)
    preview_bgr = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)

    return PredictionResult(
        filename=image_path.name,
        predicted_class=CLASS_NAMES[pred_idx],
        confidence=float(probs[pred_idx]),
        probabilities={name: float(prob) for name, prob in zip(CLASS_NAMES, probs)},
        preview_image=image_to_data_url(preview_bgr),
        gradcam=image_to_data_url(gradcam),
        gradcampp=image_to_data_url(gradcampp),
        scorecam=image_to_data_url(scorecam),
    )
