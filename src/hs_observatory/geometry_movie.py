"""Animate saved mission CDF geometry using the existing static figure variants.

This module does not import observation generation or field evaluation code.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from hs_observatory.io.cdf_positions import (
    FILL_VALUE_THRESHOLD,
    read_cdf_positions,
    select_cdf_record,
)

KINDS = {'overview': 'summary', 'polyhedra': 'polyhedra', 'formation': 'formation'}
INSTALL_HELP = "Movie encoding requires the movie extra: pip install -e '.[movie]'"


def utc_time(value: str) -> datetime:
    """Parse ISO time, interpreting naive inputs as UTC."""
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f'Invalid ISO timestamp: {value}') from exc


def frame_times(start: str, end: str, cadence: float) -> list[datetime]:
    """Select regular requested times through end, without an irregular final step."""
    first, last = utc_time(start), utc_time(end)
    if last < first:
        raise ValueError('end must not be before start')
    if not np.isfinite(cadence) or cadence < 1 or not float(cadence).is_integer():
        raise ValueError('cadence must be an integer multiple of the native 1-hour cadence (minimum 1 hour)')
    # Avoid overflowing timedelta for an enormous cadence on a short interval.
    count = int((last - first).total_seconds() // (cadence * 3600)) + 1
    if count == 1:
        return [first]
    step = timedelta(hours=int(cadence))
    return [first + index * step for index in range(count)]


def expand_kinds(kinds):
    if not kinds or any(kind not in (*KINDS, 'all') for kind in kinds):
        raise ValueError('kinds must be overview, polyhedra, formation, or all')
    return list(KINDS) if 'all' in kinds else list(dict.fromkeys(kinds))


def time_label(value: datetime) -> str:
    return value.replace(tzinfo=None).isoformat()


def time_tag(value: datetime) -> str:
    tag = value.strftime('%Y%m%dT%H%M%S')
    return tag + (f'{value.microsecond:06d}' if value.microsecond else '') + 'Z'


def movie_stem(start: str, end: str, cadence: float) -> str:
    return f'{time_tag(utc_time(start))}_{time_tag(utc_time(end))}_{int(cadence)}h'


def load_frames(cdf_path: Path, times: list[datetime]):
    """Read once, then reuse the static loader's nearest-record selection."""
    epochs, positions, labels, sources = read_cdf_positions(cdf_path)
    if (len(epochs) == 0 or np.any(np.isnat(epochs)) or np.any(np.diff(epochs) < np.timedelta64(0, 's'))
            or positions.shape != (len(epochs), len(labels), 3) or len(sources) != len(epochs)):
        raise ValueError('CDF geometry has invalid epochs, positions, or source records')
    requested = [np.datetime64(time_label(time), 'ns') for time in times]
    if any(time < epochs[0] or time > epochs[-1] for time in requested):
        raise ValueError(f'Requested frame times are outside CDF coverage: {epochs[0]} to {epochs[-1]} UTC')
    frames = []
    ordered_labels = None
    ordered_indices = None
    for time in times:
        sample = select_cdf_record(epochs, positions, labels, sources, requested_time=time_label(time))
        valid = sample['valid'] & np.all(np.isfinite(sample['selected_position']), axis=1)
        indices = np.flatnonzero(valid).tolist()
        ids = [labels[index] for index in indices]
        if len(ids) != 9 or len(set(ids)) != 9 or 'H' not in ids:
            raise ValueError(f'{sample["selected_time"]}: geometry requires nine valid spacecraft including hub H')
        hub = indices.pop(ids.index('H'))
        indices.insert(0, hub)
        ids = [labels[index] for index in indices]
        if ordered_labels is not None and ids != ordered_labels:
            raise ValueError('Valid spacecraft identities change between selected records')
        ordered_labels, ordered_indices = ids, indices
        frames.append({'requested_time': sample['requested_time'],
                       'selected_time': sample['selected_time'],
                       'selected_index': sample['selected_index'],
                       'source_file': sample['source_file'],
                       'position': sample['selected_position'][indices]})
    context = positions[:, ordered_indices, :]
    valid_context = np.all(np.isfinite(context) & (context > FILL_VALUE_THRESHOLD), axis=(1, 2))
    return frames, ordered_labels, context[valid_context].swapaxes(0, 1)


def render_frame(frame, spacecraft_ids, context, kind, path):
    """Render a fixed canvas with the static plot's styling and timestamp metadata."""
    import matplotlib.pyplot as plt
    from hs_observatory.visualization import plot_geometry_summary

    metadata = {key: frame[key] for key in ('requested_time', 'selected_time', 'source_file')}
    metadata['source'] = 'cdf'
    figure, _ = plot_geometry_summary(frame['position'], spacecraft_ids,
        metadata=metadata, context_trajectory=context, variant=KINDS[kind])
    try:
        if np.datetime64(frame['requested_time']) != np.datetime64(frame['selected_time']):
            # The static renderer already labels the selected epoch. Append the
            # requested epoch to its header without altering panel styling.
            requested = f'Requested: {frame["requested_time"]} UTC'
            if kind == 'overview':
                title = figure._suptitle
                title.set_text(title.get_text() + ' | ' + requested)
            else:
                epoch = next(text for text in figure.texts if text.get_text().startswith('Epoch:'))
                epoch.set_text(epoch.get_text() + ' | ' + requested)
        path.parent.mkdir(parents=True, exist_ok=True)
        with plt.rc_context({'savefig.bbox': None}):
            figure.savefig(path, dpi=150, bbox_inches=None, facecolor='white', transparent=False)
    finally:
        plt.close(figure)


def movie_dependencies(output_format):
    try:
        import imageio.v2 as imageio
        if output_format == 'mp4':
            import imageio_ffmpeg
            imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(INSTALL_HELP + f' ({exc})') from exc
    return imageio


def assemble_animation(paths, destination: Path, fps: float):
    """Append ordered PNGs and atomically publish a successfully closed animation."""
    if not paths:
        raise ValueError('Cannot assemble an animation without frames')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and greater than zero')
    output_format = destination.suffix.lstrip('.')
    if output_format not in ('gif', 'mp4'):
        raise ValueError('Animation format must be gif or mp4')
    imageio = movie_dependencies(output_format)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.geometry-encode-', dir=destination.parent) as staging:
        temporary = Path(staging) / destination.name
        options = ({'format': 'GIF', 'duration': 1000 / fps, 'loop': 0}
                   if output_format == 'gif' else
                   {'format': 'FFMPEG', 'fps': fps, 'codec': 'libx264',
                    'pixelformat': 'yuv420p', 'macro_block_size': 1, 'ffmpeg_log_level': 'error'})
        shape = None
        try:
            with imageio.get_writer(temporary, mode='I', **options) as writer:
                for path in paths:
                    pixels = imageio.imread(path)[..., :3]
                    if shape is not None and pixels.shape != shape:
                        raise ValueError('Animation frames must have identical canvas dimensions')
                    shape = pixels.shape
                    if output_format == 'mp4':
                        pixels = np.pad(pixels, ((0, pixels.shape[0] % 2),
                                                 (0, pixels.shape[1] % 2), (0, 0)), mode='edge')
                    writer.append_data(pixels)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError('Encoder did not produce an animation')
            temporary.replace(destination)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(f'Could not encode {output_format} animation: {exc}. {INSTALL_HELP}') from exc


def make_movie(args):
    times = frame_times(args.start, args.end, args.cadence)
    if not np.isfinite(args.fps) or args.fps <= 0:
        raise ValueError('fps must be finite and greater than zero')
    kinds = expand_kinds(args.kinds)
    frames, spacecraft_ids, context = load_frames(args.cdf_path, times)
    stem = movie_stem(args.start, args.end, args.cadence)
    outputs = {kind: args.output_dir / f'geometry_{kind}_{stem}.{args.format}' for kind in kinds}
    for index, frame in enumerate(frames):
        print(f'{index:06d}: requested {frame["requested_time"]} UTC -> selected {frame["selected_time"]} UTC',
              flush=True)
    for path in outputs.values():
        print(f'Output: {path}', flush=True)
    if args.dry_run:
        return list(outputs.values())
    movie_dependencies(args.format)  # Fail before rendering if the extra/encoder is missing.
    import matplotlib
    matplotlib.use('Agg')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.geometry-frames-', dir=args.output_dir))
    try:
        manifest = {'cdf_path': str(args.cdf_path), 'start': time_label(utc_time(args.start)),
                    'end': time_label(utc_time(args.end)), 'cadence_hours': int(args.cadence),
                    'fps': args.fps, 'format': args.format, 'kinds': kinds,
                    'spacecraft_ids': spacecraft_ids,
                    'frames': [{key: value for key, value in frame.items() if key != 'position'}
                               | {'index': index, 'filename': f'frame_{index:06d}.png'}
                               for index, frame in enumerate(frames)]}
        (staging / 'frames.json').write_text(json.dumps(manifest, indent=2) + '\n')
        for kind, output in outputs.items():
            paths = []
            for index, frame in enumerate(frames):
                path = staging / kind / f'frame_{index:06d}.png'
                render_frame(frame, spacecraft_ids, context, kind, path)
                paths.append(path)
            assemble_animation(paths, output, args.fps)
    finally:
        if args.keep_frames:
            # Allocate a unique destination so retained frames never overwrite a prior run.
            retained = Path(tempfile.mkdtemp(prefix=f'geometry_frames_{stem}_', dir=args.output_dir))
            retained.rmdir()
            staging.rename(retained)
            print(f'Frames: {retained}', flush=True)
        else:
            shutil.rmtree(staging)
    return list(outputs.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description='Visualize observatory geometry without generating observations')
    commands = parser.add_subparsers(dest='command', required=True)
    movie = commands.add_parser('geometry-movie', help='Animate native CDF formation snapshots')
    movie.add_argument('--cdf-path', type=Path, required=True, help='Mission CDF file or directory')
    movie.add_argument('--start', required=True, help='ISO start timestamp (UTC if timezone omitted)')
    movie.add_argument('--end', required=True, help='ISO end timestamp (inclusive when cadence-aligned)')
    movie.add_argument('--cadence', type=float, required=True, metavar='HOURS', help='Integer hours, minimum 1')
    movie.add_argument('--kinds', nargs='+', choices=(*KINDS, 'all'), default=['all'])
    movie.add_argument('--format', choices=('gif', 'mp4'), default='gif')
    movie.add_argument('--output-dir', type=Path, required=True)
    movie.add_argument('--fps', type=float, default=5.)
    movie.add_argument('--keep-frames', action='store_true')
    movie.add_argument('--dry-run', action='store_true', help='Validate and list frames without rendering/encoding')
    args = parser.parse_args(argv)
    try:
        make_movie(args)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(2, f'error: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
