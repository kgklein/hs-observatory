#!/usr/bin/env python3
"""Generate malara-ts duration cases in pairs, longest first.

Usage:
    .venv/bin/python scripts/generate_malara_timeseries.py --dry-run
    .venv/bin/python scripts/generate_malara_timeseries.py

Existing HDF5 outputs are skipped unless --overwrite is supplied. Both jobs in
one pair finish before the next pair starts. A failed pair stops the batch.
"""

import argparse
import os
from pathlib import Path
import signal
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]


def discover_cases(root, overwrite=False):
    """Read durations and output paths from recipes, without modifying them."""
    cases = []
    for path in sorted((root / 'generation').glob('malara-ts-*-t*.yaml')):
        config = yaml.safe_load(path.read_text())
        duration = float(config['time']['t_stop_s']) - float(config['time']['t_start_s'])
        if duration <= 0:
            raise ValueError(f'{path}: nonpositive duration')
        output = root / config['output']['path']
        if output.exists() and not overwrite:
            print(f'Skip {path.name}: output already exists ({output})', flush=True)
            continue
        cases.append((duration, path))
    return sorted(cases, key=lambda case: (-case[0], case[1].name))


def run_pair(cases, root, log_dir):
    """Wait for the whole pair; terminate worker process groups on interruption."""
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MPLBACKEND='Agg')
    env['PYTHONPATH'] = str(root / 'src') + (
        os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    jobs = []
    logs = []
    try:
        for duration, config in cases:
            log_path = log_dir / f'{config.stem}.log'
            log = log_path.open('w')
            logs.append(log)
            process = subprocess.Popen(
                [str(root / '.venv/bin/hsobs-generate'), str(config)],
                cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            jobs.append((config, process))
            print(f'Start {config.name}: {duration:g} s; PID {process.pid}; '
                  f'log {log_path}', flush=True)
        failed = False
        for config, process in jobs:
            status = process.wait()
            failed |= status != 0
            print(f'{"FAILED" if status else "Done"} {config.name} '
                  f'(exit {status})', flush=True)
        return not failed
    finally:
        # Includes the generator's ProcessPoolExecutor children, not just its parent.
        for _, process in jobs:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for _, process in jobs:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for log in logs:
            log.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='print pairs without generating')
    parser.add_argument('--overwrite', action='store_true', help='include existing HDF5 outputs')
    parser.add_argument('--log-dir', type=Path, default=Path('logs/malara-timeseries'))
    args = parser.parse_args(argv)
    cases = discover_cases(ROOT, args.overwrite)
    if not cases:
        print('No pending duration cases.')
        return 0
    log_dir = (ROOT / args.log_dir).resolve()
    if not args.dry_run:
        if not (ROOT / '.venv/bin/hsobs-generate').is_file():
            parser.error('missing .venv/bin/hsobs-generate')
        log_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(cases), 2):
        pair = cases[start:start + 2]
        print(f'Pair {start // 2 + 1}: ' + ', '.join(path.name for _, path in pair), flush=True)
        if not args.dry_run and not run_pair(pair, ROOT, log_dir):
            print('Batch stopped after failed pair. Check the logs before restarting.', flush=True)
            return 1
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted; active generator process groups stopped.', flush=True)
        raise SystemExit(130) from None
