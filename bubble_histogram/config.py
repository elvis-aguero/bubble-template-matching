from dataclasses import dataclass


@dataclass
class PipelineConfig:
    num_templates: int = 1
    template_size: int = 10
    scale_factor: float = 0.9
    min_radius: float = 1.0
    max_radius: float = 50.0
    n_score_bins: int = 50
    neg_sample_ratio: int = 10
    min_neg_dist: int = 10
    local_maxima_calibration: bool = False  # sample f⁻ from local maxima and use LM prior during calibration
    predict_local_maxima: bool = False      # sum P over local maxima only (not all pixels) during predict()
