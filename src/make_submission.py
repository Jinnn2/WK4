from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ID_COL, TARGET_COL


def postprocess_predictions(pred: np.ndarray, round_to_int: bool = True) -> np.ndarray:
    clipped = np.maximum(pred, 0)
    if round_to_int:
        return np.round(clipped).astype(int)
    return clipped


def save_submission(
    test_ids: pd.Series,
    pred: np.ndarray,
    output_path: Path,
    round_to_int: bool = True,
) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(
        {
            ID_COL: test_ids.to_numpy(),
            TARGET_COL: postprocess_predictions(pred, round_to_int=round_to_int),
        }
    )
    submission.to_csv(output_path, index=False)
    return submission

