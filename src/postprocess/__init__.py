from .peak_extract import extract_peak_mask, peak_mask_to_markers, smooth_heatmap
from .refine import fill_unassigned_pixels, filter_small_regions, relabel_sequential, split_disconnected_regions
from .watershed import watershed_from_markers


def run_postprocess(
    interior_pred,
    seed_pred,
    interior_threshold: float = 0.5,
    smooth_kernel_size: int = 7,
    smooth_sigma: float = 1.5,
    peak_threshold: float = 0.35,
    min_distance: int = 5,
    min_area: int = 16,
):
    """完整后处理流程：平滑 heatmap → 提峰值 → watershed → refine。"""
    interior_bin = (interior_pred >= interior_threshold).to(seed_pred.dtype)
    seed_smooth = smooth_heatmap(seed_pred, kernel_size=smooth_kernel_size, sigma=smooth_sigma)
    peak_mask = extract_peak_mask(
        seed_heatmap=seed_smooth,
        interior_bin=interior_bin,
        peak_threshold=peak_threshold,
        min_distance=min_distance,
    )
    markers = peak_mask_to_markers(peak_mask)
    label_map = watershed_from_markers(interior_bin=interior_bin, markers=markers)
    label_map = fill_unassigned_pixels(label_map, interior_bin)
    label_map = split_disconnected_regions(label_map)
    label_map = filter_small_regions(label_map, min_area=min_area)
    label_map = relabel_sequential(label_map)

    return {
        "interior_bin": interior_bin,
        "seed_smooth": seed_smooth,
        "peak_mask": peak_mask,
        "markers": markers,
        "label_map": label_map,
    }


__all__ = [
    "extract_peak_mask",
    "peak_mask_to_markers",
    "smooth_heatmap",
    "watershed_from_markers",
    "fill_unassigned_pixels",
    "filter_small_regions",
    "relabel_sequential",
    "split_disconnected_regions",
    "run_postprocess",
]
