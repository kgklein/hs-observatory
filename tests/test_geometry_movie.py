"""Geometry-only movie tests using fabricated snapshots and tiny image frames."""

from argparse import Namespace
import builtins
from datetime import timezone
import json
from pathlib import Path
from unittest.mock import Mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

from hs_observatory import geometry_movie as movie
from hs_observatory.io import cdf_positions
from hs_observatory.spacecraft.configuration import synthetic_configuration


@pytest.fixture
def geometry(monkeypatch):
    epochs = np.arange('2030-08-25T00', '2030-08-25T05', dtype='datetime64[h]').astype('datetime64[ns]')
    spacecraft = synthetic_configuration(np.array([0.]))
    positions = np.repeat(spacecraft.position[:, 0][None], len(epochs), axis=0)
    positions += np.arange(len(epochs))[:, None, None] * 10.
    # Hub is deliberately not the first CDF label.
    order = [1, 2, 0, 3, 4, 5, 6, 7, 8]
    labels = [spacecraft.spacecraft_ids[index] for index in order]
    data = epochs, positions[:, order], labels, ['mission.cdf'] * len(epochs)
    reader = Mock(return_value=data)
    monkeypatch.setattr(movie, 'read_cdf_positions', reader)
    return data, reader


def options(tmp_path, **kwargs):
    values = dict(cdf_path=Path('mission.cdf'), start='2030-08-25T00:00:00',
                  end='2030-08-25T02:00:00', cadence=1., kinds=['all'], format='gif',
                  output_dir=tmp_path / 'movies', fps=5., keep_frames=False, dry_run=False)
    return Namespace(**(values | kwargs))


def fake_renderer(frame, spacecraft_ids, context, kind, path, *, axis_limits=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (8, 6), (frame['selected_index'] * 50, 0, 0)).save(path)


def fake_encoder(paths, destination, fps):
    destination.write_bytes(b'animation')


def test_time_sequence_and_names():
    times = movie.frame_times('2030-08-25T00:00:00', '2030-08-25T04:00:00', 2)
    assert [time.hour for time in times] == [0, 2, 4]
    assert all(time.tzinfo == timezone.utc for time in times)
    assert len(movie.frame_times('2030-08-25', '2030-08-25', 1)) == 1
    assert [time.hour for time in movie.frame_times('2030-08-25', '2030-08-25T03:00', 2)] == [0, 2]
    assert movie.utc_time('2030-08-25T02:00:00+02:00') == times[0]
    assert movie.utc_time('2030-08-25T00:00:00Z') == times[0]
    assert movie.movie_stem('2030-08-25T02:00+02:00', '2030-08-25T04:00Z', 2.) == (
        '20300825T000000Z_20300825T040000Z_2h')
    assert movie.expand_kinds(['formation', 'overview', 'formation']) == ['formation', 'overview']
    assert movie.expand_kinds(['all']) == list(movie.KINDS)
    with pytest.raises(ValueError, match='kinds'):
        movie.expand_kinds(['other'])


@pytest.mark.parametrize('cadence', [0, -.5, .5, 1.5, float('nan'), float('inf')])
def test_invalid_cadence(cadence):
    with pytest.raises(ValueError, match='1-hour'):
        movie.frame_times('2030-08-25', '2030-08-26', cadence)


def test_invalid_times():
    with pytest.raises(ValueError, match='before start'):
        movie.frame_times('2030-08-26', '2030-08-25', 1)
    with pytest.raises(ValueError, match='ISO timestamp'):
        movie.frame_times('invalid', '2030-08-26', 1)


def test_load_once_nearest_ties_repeats_and_hub_order(geometry):
    _, reader = geometry
    times = [movie.utc_time(value) for value in ('2030-08-25T00:20', '2030-08-25T00:30', '2030-08-25T00:40')]
    frames, labels, context = movie.load_frames(Path('mission.cdf'), times)
    reader.assert_called_once()
    assert [frame['selected_index'] for frame in frames] == [0, 0, 1]
    assert labels[0] == 'H'
    assert context.shape == (9, 5, 3)
    np.testing.assert_array_equal(frames[0]['position'], context[:, 0])


def test_static_selector_preserved(geometry, monkeypatch):
    data, _ = geometry
    monkeypatch.setattr(cdf_positions, 'read_cdf_positions', Mock(return_value=data))
    selected = cdf_positions.select_cdf_configuration_sample('mission.cdf', requested_time='2030-08-25T00:30')
    assert selected['selected_index'] == 0
    np.testing.assert_array_equal(selected['selected_position'], data[1][0])
    # Existing static behavior still permits selecting a nearest endpoint outside coverage.
    assert cdf_positions.select_cdf_configuration_sample('mission.cdf', requested_time='2030-08-24')['selected_index'] == 0
    assert cdf_positions.select_cdf_configuration_sample('mission.cdf')['selected_index'] == 0


@pytest.mark.parametrize('problem', ['outside', 'fill', 'nan', 'hub', 'missing', 'all_invalid'])
def test_invalid_geometry(geometry, problem):
    data, reader = geometry
    epochs, positions, labels, sources = data
    time = '2030-08-25T00:00'
    if problem == 'outside':
        time = '2030-08-24T23:00'
    elif problem == 'fill':
        positions[0, 0] = -1e31
    elif problem == 'nan':
        positions[0, 0] = np.nan
    elif problem == 'hub':
        labels[2] = 'other'
    elif problem == 'missing':
        reader.side_effect = FileNotFoundError('missing CDF')
        with pytest.raises(FileNotFoundError):
            movie.load_frames(Path('missing'), [movie.utc_time(time)])
        return
    else:
        positions[0] = -1e31
    with pytest.raises(ValueError):
        movie.load_frames(Path('mission.cdf'), [movie.utc_time(time)])


@pytest.mark.parametrize('fps', [0, -1, float('inf'), float('nan')])
def test_invalid_fps(tmp_path, fps):
    with pytest.raises(ValueError, match='fps'):
        movie.make_movie(options(tmp_path, fps=fps))


def test_dry_run_has_no_render_or_encoder_dependency(geometry, tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail('dry-run rendered or imported encoding dependencies')
    monkeypatch.setattr(movie, 'movie_dependencies', forbidden)
    monkeypatch.setattr(movie, 'render_frame', forbidden)
    monkeypatch.setattr(movie, 'movie_axis_limits', forbidden)
    outputs = movie.make_movie(options(tmp_path, dry_run=True, start='2030-08-25T00:20', end='2030-08-25T01:20'))
    assert len(outputs) == 3
    assert outputs[0].name == 'geometry_overview_20300825T002000Z_20300825T012000Z_1h.gif'
    assert not (tmp_path / 'movies').exists()
    assert 'requested 2030-08-25T00:20:00 UTC -> selected 2030-08-25T00:00:00' in capsys.readouterr().out


@pytest.mark.parametrize('keep', [False, True])
def test_frame_enumeration_retention_and_output_kinds(geometry, tmp_path, monkeypatch, keep):
    monkeypatch.setattr(movie, 'movie_dependencies', Mock())
    renderer = Mock(side_effect=fake_renderer)
    encoder = Mock(side_effect=fake_encoder)
    monkeypatch.setattr(movie, 'render_frame', renderer)
    monkeypatch.setattr(movie, 'assemble_animation', encoder)
    scan = Mock(wraps=movie.movie_axis_limits)
    monkeypatch.setattr(movie, 'movie_axis_limits', scan)
    args = options(tmp_path, keep_frames=keep, fps=7.)
    outputs = movie.make_movie(args)
    assert len(outputs) == 3 and all(path.read_bytes() == b'animation' for path in outputs)
    assert renderer.call_count == 9
    scan.assert_called_once()
    fixed = renderer.call_args_list[0].kwargs['axis_limits']
    assert all(call.kwargs['axis_limits'] is fixed for call in renderer.call_args_list)
    assert encoder.call_count == 3
    for call in encoder.call_args_list:
        assert [path.name for path in call.args[0]] == [f'frame_{index:06d}.png' for index in range(3)]
        assert call.args[2] == 7.
    assert not list(args.output_dir.glob('.geometry-frames-*'))
    retained = list(args.output_dir.glob('geometry_frames_*'))
    assert len(retained) == int(keep)
    if keep:
        manifest = json.loads((retained[0] / 'frames.json').read_text())
        assert manifest['axis_limits']['spatial_extent'] == fixed.spatial_extent
        assert manifest['axis_limits']['baseline_colors'] == list(fixed.baseline_colors)
        assert manifest['spacecraft_ids'][0] == 'H'
        assert [frame['index'] for frame in manifest['frames']] == [0, 1, 2]
        assert manifest['frames'][0]['filename'] == 'frame_000000.png'
        for kind in movie.KINDS:
            assert len(list((retained[0] / kind).glob('*.png'))) == 3
    movie.make_movie(options(tmp_path, kinds=['formation'], keep_frames=keep))
    if keep:
        assert len(list(args.output_dir.glob('geometry_frames_*'))) == 2


@pytest.mark.parametrize('keep', [False, True])
def test_failure_cleanup(geometry, tmp_path, monkeypatch, keep):
    monkeypatch.setattr(movie, 'movie_dependencies', Mock())
    monkeypatch.setattr(movie, 'render_frame', fake_renderer)
    monkeypatch.setattr(movie, 'assemble_animation', Mock(side_effect=RuntimeError('encoder failed')))
    args = options(tmp_path, keep_frames=keep)
    with pytest.raises(RuntimeError, match='encoder failed'):
        movie.make_movie(args)
    assert not list(args.output_dir.glob('.geometry-frames-*'))
    assert len(list(args.output_dir.glob('geometry_frames_*'))) == int(keep)


@pytest.mark.parametrize('kind, variant', list(movie.KINDS.items()))
def test_render_dispatch_fixed_canvas_and_timestamp(geometry, tmp_path, monkeypatch, kind, variant):
    from hs_observatory import visualization

    frames, ids, context = movie.load_frames(Path('mission.cdf'), [movie.utc_time('2030-08-25T00:20')])
    figure = plt.figure(figsize=(4, 3))
    figure.suptitle('Selected epoch: 2030-08-25T00:00:00')
    figure.text(.1, .9, 'Epoch: 2030-08-25T00:00:00 UTC')
    render = Mock(return_value=(figure, {}))
    monkeypatch.setattr(visualization, 'plot_geometry_summary', render)
    path = tmp_path / 'kind' / 'frame_000000.png'
    with plt.rc_context({'savefig.bbox': 'tight'}):
        movie.render_frame(frames[0], ids, context, kind, path)
    assert Image.open(path).size == (600, 450)
    assert not plt.fignum_exists(figure.number)
    call = render.call_args
    assert call.kwargs['variant'] == variant
    assert call.kwargs['metadata']['selected_time'].startswith('2030-08-25T00:00:00')
    assert any('Requested: 2030-08-25T00:20:00' in text.get_text() for text in figure.texts)


def test_actual_static_renderer_smoke(geometry, tmp_path):
    frames, ids, context = movie.load_frames(Path('mission.cdf'), [movie.utc_time('2030-08-25T00:00')])
    movie.render_frame(frames[0], ids, context, 'overview', tmp_path / 'overview.png')
    with Image.open(tmp_path / 'overview.png') as image:
        assert image.size == (4650, 2400)
    assert not plt.get_fignums()


@pytest.fixture
def pngs(tmp_path):
    paths = []
    for index, color in enumerate(('red', 'green', 'blue')):
        path = tmp_path / f'frame_{index:06d}.png'
        Image.new('RGB', (7, 5), color).save(path)
        paths.append(path)
    return paths


def test_real_tiny_gif(pngs, tmp_path):
    pytest.importorskip('imageio')
    destination = tmp_path / 'animation.gif'
    movie.assemble_animation(pngs, destination, 5.)
    with Image.open(destination) as animation:
        assert animation.n_frames == 3
        assert animation.info['loop'] == 0
        assert animation.info['duration'] == 200
        for index, path in enumerate(pngs):
            animation.seek(index)
            np.testing.assert_array_equal(np.asarray(animation.convert('RGB')), np.asarray(Image.open(path)))


def test_mp4_writer_settings_padding_and_atomicity(pngs, tmp_path, monkeypatch):
    imageio = Namespace(imread=lambda path: np.asarray(Image.open(path)), get_writer=None)
    frames = []
    writer = Mock()
    writer.__enter__ = Mock(return_value=writer)
    writer.__exit__ = Mock(return_value=False)
    writer.append_data.side_effect = lambda pixels: frames.append(pixels.copy())
    def get_writer(path, **kwargs):
        Path(path).write_bytes(b'encoded')
        return writer
    get = Mock(side_effect=get_writer)
    monkeypatch.setattr(imageio, 'get_writer', get)
    monkeypatch.setattr(movie, 'movie_dependencies', Mock(return_value=imageio))
    destination = tmp_path / 'animation.mp4'
    movie.assemble_animation(pngs, destination, 8.)
    assert destination.read_bytes() == b'encoded'
    settings = get.call_args.kwargs
    assert settings['fps'] == 8. and settings['codec'] == 'libx264'
    assert settings['macro_block_size'] == 1
    assert [frame.shape for frame in frames] == [(6, 8, 3)] * 3
    for path, frame in zip(pngs, frames):
        np.testing.assert_array_equal(frame[:5, :7], np.asarray(Image.open(path)))
        np.testing.assert_array_equal(frame[-1], frame[-2])
        np.testing.assert_array_equal(frame[:, -1], frame[:, -2])
    destination.write_bytes(b'previous animation')
    writer.append_data.side_effect = RuntimeError('encoder failed')
    with pytest.raises(RuntimeError, match='Could not encode mp4'):
        movie.assemble_animation(pngs, destination, 8.)
    assert destination.read_bytes() == b'previous animation'
    assert not list(tmp_path.glob('.geometry-encode-*'))


def test_missing_movie_extra(monkeypatch):
    original_import = builtins.__import__
    def missing(name, *args, **kwargs):
        if name.startswith('imageio'):
            raise ImportError('missing imageio')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', missing)
    with pytest.raises(RuntimeError, match=r'pip install.*movie'):
        movie.movie_dependencies('gif')


def test_cli_dispatch_and_errors(geometry, tmp_path, monkeypatch, capsys):
    args = ['geometry-movie', '--cdf-path', 'mission.cdf', '--start', '2030-08-25',
            '--end', '2030-08-25T02:00', '--cadence', '1', '--output-dir', str(tmp_path / 'movie')]
    assert movie.main(args + ['--dry-run']) == 0
    with pytest.raises(SystemExit) as error:
        movie.main(args + ['--cadence', '.5'])
    assert error.value.code == 2
    assert '1-hour' in capsys.readouterr().err
    def forbidden(*args, **kwargs):
        pytest.fail('dry-run loaded visualization or generation code')
    monkeypatch.setattr(movie, 'render_frame', forbidden)
    assert movie.main(args + ['--dry-run', '--format', 'mp4', '--kinds', 'overview', 'formation']) == 0
