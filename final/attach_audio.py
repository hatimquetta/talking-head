#!/usr/bin/env python3
"""Milestone 3 - attach TTS audio to a sequenced idle -> speaking -> idle video.

The audio is placed starting at OFFSET (default 3.0s) so it sits exactly over
the speaking segment. Depending on --mode it writes:

  mux       one .mp4 with the audio baked in at OFFSET   ({stem}_with_audio.mp4)
  separate  a standalone audio padded with OFFSET seconds of silence at the
            front and back, so it is the SAME length as the video and lines up
            1:1 (kept as a separate asset)              ({stem}_audio_padded.m4a)
  both      both of the above (default)

Usage
-----
    python attach_audio.py VIDEO AUDIO [--offset 3.0] [--mode both] [-o OUTDIR]

Requires ffmpeg + ffprobe on PATH.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# Default output location: <this script's folder>/output/with_audio/
FINAL_DIR = Path(__file__).resolve().parent / 'output' / 'with_audio'


def _dur(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                          '-of', 'json', str(path)], capture_output=True, text=True).stdout
    return float(json.loads(out)['format']['duration'])


def _ff(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('ffmpeg failed:\n' + (r.stderr or '')[-1500:])


def mux(video, audio, out, offset):
    """Video copied as-is; audio delayed to start at `offset`."""
    delay_ms = int(round(offset * 1000))
    _ff(['ffmpeg', '-y', '-i', str(video), '-i', str(audio),
         '-filter_complex', f'[1:a]adelay=delays={delay_ms}:all=1[a]',
         '-map', '0:v', '-map', '[a]',
         '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', str(out)])
    return out


def pad_separate(audio, out, offset, total):
    """Audio with `offset`s silence front + back, trimmed to `total` seconds."""
    delay_ms = int(round(offset * 1000))
    _ff(['ffmpeg', '-y', '-i', str(audio),
         '-af', f'adelay={delay_ms}:all=1,apad', '-t', f'{total:.3f}',
         '-c:a', 'aac', '-b:a', '192k', str(out)])
    return out


def main():
    p = argparse.ArgumentParser(
        description='Attach TTS audio to a sequenced idle->speaking->idle video.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('video', help='sequenced idle+speaking+idle video (silent)')
    p.add_argument('audio', help='TTS audio to place over the speaking segment')
    p.add_argument('--offset', type=float, default=3.0,
                   help='seconds before the audio starts (= idle lead-in, default 3.0)')
    p.add_argument('--mode', choices=['mux', 'separate', 'both'], default='both',
                   help='what to produce (default: both)')
    p.add_argument('--output', '-o', default=None,
                   help='output directory (default: output/with_audio)')
    args = p.parse_args()

    if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:
        sys.exit('ERROR: ffmpeg and ffprobe must be installed and on PATH.')
    for f in (args.video, args.audio):
        if not os.path.exists(f):
            sys.exit(f'ERROR: input not found: {f}')

    stem = Path(args.video).stem
    out_dir = args.output or str(FINAL_DIR)
    os.makedirs(out_dir, exist_ok=True)
    total = _dur(args.video)
    adur = _dur(args.audio)

    print(f'video : {total:.2f}s   audio : {adur:.2f}s   offset : {args.offset:.1f}s')
    speaking_window = total - 2 * args.offset
    if args.offset + adur > total + 0.05:
        print(f'  NOTE: audio ends at {args.offset + adur:.2f}s, past the video ({total:.2f}s) - '
              f'it will be cut. Rebuild the video with S = {adur:.2f}s (sequencer.py).')
    else:
        print(f'  speaking window S = {speaking_window:.2f}s ; audio = {adur:.2f}s '
              f'({"fits" if adur <= speaking_window + 0.05 else "over the window"})')

    if args.mode in ('mux', 'both'):
        out = os.path.join(out_dir, f'{stem}_with_audio.mp4')
        mux(args.video, args.audio, out, args.offset)
        print(f'  muxed    -> {out}')
    if args.mode in ('separate', 'both'):
        out = os.path.join(out_dir, f'{stem}_audio_padded.m4a')
        pad_separate(args.audio, out, args.offset, total)
        print(f'  padded   -> {out}  ({_dur(out):.2f}s, matches the video)')


if __name__ == '__main__':
    main()
