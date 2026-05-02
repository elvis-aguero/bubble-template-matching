from dataclasses import dataclass


@dataclass
class PipelineConfig:
    num_templates: int = 1              # how many distinct appearance templates to build; 1 pools all bubble sizes together
    template_size: int = 13             # side length (px) of the square template; every bubble patch is resized to this
    scale_factor: float = 0.9          # image is multiplied by this at each pyramid level; controls size-bin width
    min_radius: float = 3.0            # smallest bubble radius (in original image px) the pyramid covers; radii below canonical_radius require upscaling levels
    max_radius: float = 50.0           # largest bubble radius (in original image px) the pyramid covers
    n_score_bins: int = 50             # number of histogram bins for the P(score|bubble) and P(score|bg) lookup tables
    neg_sample_ratio: int = 10         # how many random non-bubble pixels to sample per annotated bubble during calibration
    min_neg_dist: int = 10             # pixels — a random pixel must be at least this far from any bubble to count as negative
    template_context_factor: float = 1.3   # crop box = 2*(r*factor) × 2*(r*factor); >1.0 adds a background ring around the bubble
    local_maxima_calibration: bool = True  # if True, train calibrator on local-maxima scores instead of random pixel scores
    predict_local_maxima: bool = True       # if True, sum P(bubble) only at local maxima; if False, sum over every pixel (overcounts)
    nms_iou_threshold: float = 0.5         # IoU threshold for cross-scale NMS in predict(); 0.0 disables cross-scale suppression
    nms_max_candidates: int = 50000       # global cap after per-level filtering; set large so the per-level cap dominates
    nms_max_candidates_per_level: int = 1000  # per-level cap: includes ~top-30% of level LMs so GT bubble peaks survive; limits NMS input to ~27K
