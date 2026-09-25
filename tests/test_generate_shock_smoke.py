import json
from pathlib import Path

import h5py
import numpy as np
import yaml

from hs_observatory.cli import generate_main


def test_shock_example_cli_hdf5_smoke(tmp_path, capsys):
    example = Path(__file__).parents[1] / "configs" / "example_generate_shock.yaml"
    config = yaml.safe_load(example.read_text(encoding="utf-8"))
    output = tmp_path / "shock.h5"
    config["spacecraft"] = {
        "source": "synthetic",
        "velocity_km_s": config["spacecraft"]["velocity_km_s"],
    }
    config["output"]["path"] = str(output)
    config["visualization"]["enabled"] = False
    temporary_config = tmp_path / "example_generate_shock.yaml"
    temporary_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    assert generate_main([str(temporary_config)]) == 0
    assert str(output) in capsys.readouterr().out

    with h5py.File(output) as handle:
        assert handle["truth/n"].shape == (321, 9)
        assert handle["truth/v"].shape == (321, 9, 3)
        assert handle["truth/v_spacecraft_frame"].shape == (321, 9, 3)
        assert handle["truth/B"].shape == (1281, 9, 3)
        assert handle["observed/n"].shape == (10, 321, 9)
        assert handle["observed/v"].shape == (10, 321, 9, 3)
        assert handle["observed/B"].shape == (10, 1281, 9, 3)
        assert handle["errors/time_offset"].shape == (10, 9)
        assert handle["errors/position_offset"].shape == (10, 9, 3)
        assert handle["metadata"].attrs["field_model"] == "shock"
        assert handle["metadata"].attrs["velocity_truth_frame"] == "plasma"
        assert handle["metadata"].attrs["velocity_observed_frame"] == "spacecraft"

        metadata = json.loads(handle["metadata"].attrs["field_metadata"])
        assert metadata["reference_time_s"] == 0.0
        assert metadata["compression_ratio"] > 1.0
        assert max(metadata["jump_residuals"].values()) < 1.0e-8
        np.testing.assert_allclose(metadata["mesocenter_gse_km"], [7.0 / 3.0, 2.0, 17.0 / 9.0])

        density = handle["truth/n"][...]
        upstream = metadata["upstream"]["density_cm3"]
        downstream = metadata["downstream"]["density_cm3"]
        assert np.any(density < upstream + 0.01 * (downstream - upstream))
        assert np.any(density > upstream + 0.99 * (downstream - upstream))
