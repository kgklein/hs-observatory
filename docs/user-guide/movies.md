# Movies

`hsobs-visualize geometry-movie` animates the existing overview, polyhedra, and
formation/separation figures. It reads mission CDF positions directly and does
not generate observations, field samples, measurement errors, or trajectories.

## Install and render

Install the optional encoders and register the CLI in your environment. See
[Installation](../getting-started/installation.md#optional-dependencies) for environment
setup; mission CDF files are downloaded separately:

```bash
.venv/bin/python -m pip install -e '.[movie]'
```

From the project root:

```bash
hsobs-visualize geometry-movie \
  --cdf-path ../HelioSwarm/HS-RT/PhB_SRD8_0x78f/ \
  --start 2030-08-25T00:00:00 --end 2030-08-26T00:00:00 \
  --cadence 1 --kinds all --format gif \
  --output-dir outputs/geometry_movies --fps 5
```

## Time selection and output options

`--cadence` is in hours and must be a finite integer of at least 1. ISO timestamps
without a timezone are UTC; timezone offsets are normalized to UTC. The sequence
is `start + k * cadence <= end`: an off-cadence end does not add a shorter final
step. Equal start and end produces a single-frame animation.

The CDF file or directory is required. Each requested time selects the nearest
native record, choosing the earlier record on a tie. Requested frame times must
fall within CDF coverage. Records must contain nine valid spacecraft including
hub `H`. Repeated selections remain in the animation, preserving its timing.
The command logs requested and selected epochs; figure headers show both when
they differ. The orbit-context panel uses the full valid CDF trajectory.

`--kinds` defaults to `all`, producing three independent animations. You can also
request one kind or several, for example `--kinds overview formation`.
`--format` defaults to `gif`; `--format mp4` uses H.264 via imageio-ffmpeg.
`--fps` defaults to 5 and must be finite and positive. GIF timing is quantized by
its centisecond time resolution; MP4 is preferable for precise high frame rates.
The current static styling and 150 DPI are retained. Frames have fixed canvas
sizes; MP4 receives edge padding to even dimensions where necessary.

## Preflight and shell runner

The following preflight loads and validates geometry but neither renders figures
nor requires encoding dependencies:

```bash
hsobs-visualize geometry-movie \
  --cdf-path ../HelioSwarm/HS-RT/PhB_SRD8_0x78f/ \
  --start 2030-08-25T00:00:00 --end 2030-08-26T00:00:00 \
  --cadence 1 --output-dir outputs/geometry_movies --dry-run
```

The shell runner supplies one-hour cadence and `outputs/geometry_movies` defaults;
trailing options override them. Relative paths are resolved from your current
directory. Set `HSOBS_PYTHON` to override its Python interpreter.

```bash
bash scripts/geometry_movie.sh ../HelioSwarm/HS-RT/PhB_SRD8_0x78f/ \
  2030-08-25T00:00:00 2030-08-26T00:00:00 \
  --cadence 2 --kinds overview --format mp4 --keep-frames
```

## Animation and frame files

Outputs are named, for example,
`geometry_overview_20300825T000000Z_20300826T000000Z_2h.mp4`.
Each animation replaces any existing output of that name only after successful
encoding. Without `--keep-frames`, staging PNGs are removed on success or failure.
With it, a unique `geometry_frames_<start>_<end>_<cadence>h_*` directory is retained,
including on failure. It contains `<kind>/frame_000000.png` and subsequent frames,
plus `frames.json` mapping frame indices to requested and selected CDF records.
Partial retained frame sets may result from a failed run. Each animation is
published independently, so a completed kind remains available if a later kind fails.

## Focused validation

```bash
.venv/bin/python -m pytest -q tests/test_geometry_movie.py
.venv/bin/python -m ruff check src/hs_observatory/geometry_movie.py src/hs_observatory/io/cdf_positions.py tests/test_geometry_movie.py
bash -n scripts/geometry_movie.sh
```

The tests use fabricated geometry, a small static-renderer smoke check, tiny GIF
frames, and a mocked MP4 writer. The GIF test skips if the optional imageio package
is absent. To exercise the installed MP4 encoder without loading CDFs or rendering
mission figures, optionally run this tiny fabricated-image smoke command:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image
from hs_observatory.geometry_movie import assemble_animation
with TemporaryDirectory() as directory:
    root = Path(directory)
    frames = []
    for index, color in enumerate(('red', 'green', 'blue')):
        path = root / f'frame_{index:06d}.png'
        Image.new('RGB', (32, 24), color).save(path)
        frames.append(path)
    assemble_animation(frames, root / 'smoke.mp4', fps=5)
    print('MP4 encoder smoke passed:', (root / 'smoke.mp4').stat().st_size, 'bytes')
PY
```
