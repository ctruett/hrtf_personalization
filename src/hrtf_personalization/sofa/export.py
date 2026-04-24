from __future__ import annotations

from pathlib import Path

import numpy as np
import sofar as sf


def export_simple_free_field_hrir(
    output_path: str | Path,
    hrir: np.ndarray,
    source_positions_deg: np.ndarray,
    sampling_rate_hz: float,
) -> Path:
    """Export an HRIR tensor to a minimal SOFA file.

    Expected shapes:
    - `hrir`: (M, R, N)
    - `source_positions_deg`: (M, 3) with [azimuth, elevation, distance]
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    sofa = sf.Sofa("SimpleFreeFieldHRIR")
    sofa.Data_IR = hrir
    sofa.Data_SamplingRate = np.asarray([sampling_rate_hz], dtype=np.float64)
    sofa.SourcePosition = source_positions_deg
    sofa.SourcePosition_Type = "spherical"
    sofa.SourcePosition_Units = "degree, degree, meter"
    sofa.GLOBAL_ApplicationName = "hrtf-personalization"
    sofa.GLOBAL_ApplicationVersion = "0.1.0"
    sofa.GLOBAL_AuthorContact = "unknown"
    sf.write_sofa(output, sofa)
    return output

