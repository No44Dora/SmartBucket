from __future__ import annotations

import argparse
import functools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import local

import cv2
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image


class UnetGenerator(nn.Module):
    def __init__(self, input_nc: int, output_nc: int, num_downs: int, ngf: int = 64, norm_layer=nn.BatchNorm2d, use_dropout: bool = False):
        super().__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=None, norm_layer=norm_layer, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc: int, inner_nc: int, input_nc: int | None = None, submodule: nn.Module | None = None, outermost: bool = False, innermost: bool = False, norm_layer=nn.BatchNorm2d, use_dropout: bool = False):
        super().__init__()
        self.outermost = outermost

        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            model = [downconv, submodule, uprelu, upconv, nn.Tanh()]
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [downrelu, downconv, uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]
            if use_dropout:
                model.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)


class LineartAnimeDetector:
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = "cpu"

    @classmethod
    def from_model_path(cls, model_path: Path) -> "LineartAnimeDetector":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        net = UnetGenerator(3, 1, 8, 64, norm_layer=norm_layer, use_dropout=False)

        ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        cleaned = {}
        for key, value in ckpt.items():
            cleaned[key.replace("module.", "")] = value

        net.load_state_dict(cleaned, strict=True)
        net.eval()
        return cls(net)

    def to(self, device: str) -> "LineartAnimeDetector":
        self.model.to(device)
        self.device = device
        return self

    def __call__(self, input_image: np.ndarray, detect_resolution: int = 512) -> np.ndarray:
        input_image = hwc3(input_image)
        resized, remove_pad = resize_with_pad(input_image, detect_resolution)

        height, width, _ = resized.shape
        height_256 = 256 * int(np.ceil(float(height) / 256.0))
        width_256 = 256 * int(np.ceil(float(width) / 256.0))
        resized = cv2.resize(resized, (width_256, height_256), interpolation=cv2.INTER_CUBIC)

        with torch.no_grad():
            image_feed = torch.from_numpy(resized).float().to(self.device)
            image_feed = image_feed / 127.5 - 1.0
            image_feed = rearrange(image_feed, "h w c -> 1 c h w")

            line = self.model(image_feed)[0, 0] * 127.5 + 127.5
            line = line.clamp(0, 255).cpu().numpy().astype(np.uint8)

        detected_map = cv2.resize(hwc3(line), (width, height), interpolation=cv2.INTER_AREA)
        detected_map = remove_pad(255 - detected_map)
        return detected_map


def hwc3(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image


def resize_with_pad(image: np.ndarray, detect_resolution: int) -> tuple[np.ndarray, callable]:
    h, w, _ = image.shape
    scale = float(detect_resolution) / min(h, w)
    new_h = max(1, int(np.round(h * scale)))
    new_w = max(1, int(np.round(w * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    pad_h = (64 - new_h % 64) % 64
    pad_w = (64 - new_w % 64) % 64
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)

    def remove_pad(x: np.ndarray) -> np.ndarray:
        if pad_h > 0:
            x = x[:-pad_h, :, :]
        if pad_w > 0:
            x = x[:, :-pad_w, :]
        return x

    return padded, remove_pad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate anime lineart images with lllyasviel netG model")
    parser.add_argument("--input-dir", type=Path, default=Path("processed/color_512"))
    parser.add_argument("--output-dir", type=Path, default=Path("train/images"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--detect-resolution", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4, help="parallel worker threads for inference and saving")
    parser.add_argument("--suffixes", nargs="+", default=[".png", ".jpg", ".jpeg", ".bmp", ".webp"])
    return parser.parse_args()


def iter_images(input_dir: Path, suffixes: set[str]) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def save_lineart(detector: LineartAnimeDetector, src_path: Path, input_dir: Path, output_dir: Path, detect_resolution: int) -> None:
    rel = src_path.relative_to(input_dir)
    dst = output_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src_path) as img:
        arr = np.array(img.convert("RGB"))

    lineart = detector(arr, detect_resolution=detect_resolution)
    Image.fromarray(lineart).save(dst)


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {args.input_dir}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_images(args.input_dir, {s.lower() for s in args.suffixes})

    if not image_paths:
        print(f"[WARN] No image files found in: {args.input_dir}")
        return

    num_workers = max(1, int(args.num_workers))
    if args.device.startswith("cuda") and num_workers > 1:
        print("[WARN] CUDA mode currently forces single worker to avoid multi-thread CUDA contention.")
        num_workers = 1

    if num_workers <= 1:
        detector = LineartAnimeDetector.from_model_path(args.model_path).to(args.device)
        for src in image_paths:
            save_lineart(detector, src, args.input_dir, args.output_dir, args.detect_resolution)
    else:
        detector_cache = local()

        def _run(src: Path) -> None:
            if not hasattr(detector_cache, "detector"):
                detector_cache.detector = LineartAnimeDetector.from_model_path(args.model_path).to(args.device)
            save_lineart(detector_cache.detector, src, args.input_dir, args.output_dir, args.detect_resolution)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_run, src)
                for src in image_paths
            ]
            for future in futures:
                future.result()

    print(f"[INFO] Generated lineart for {len(image_paths)} images -> {args.output_dir} with {num_workers} worker(s)")


if __name__ == "__main__":
    main()
