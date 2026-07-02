#!/usr/bin/env python3
"""Milestone 3 - generate TTS speech audio (edge-tts: Microsoft Neural TTS, free).

Usage
-----
    python tts.py "your text here" [--gender male|female] [-o out.mp3]

Voices are Microsoft's warm, human-sounding "Multilingual" neural voices:
    female -> en-US-AvaMultilingualNeural
    male   -> en-US-AndrewMultilingualNeural
Override either with --voice <edge-tts voice id>.  (List all voices with:
`edge-tts --list-voices`.)

Prints the audio duration, which is the value you feed to sequencer.py as S.
Requires:  pip install edge-tts   (ffmpeg on PATH is only needed for the
duration readout).
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

VOICES = {
    'female': 'en-US-AvaMultilingualNeural',
    'male':   'en-US-AndrewMultilingualNeural',
}

# Default output location: <this script's folder>/output/audio/
AUDIO_DIR = Path(__file__).resolve().parent / 'output' / 'audio'


def _duration(path):
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                              '-of', 'json', str(path)], capture_output=True, text=True).stdout
        return float(json.loads(out)['format']['duration'])
    except Exception:
        return None


def generate_audio(text, out_path, voice, rate, pitch, volume):
    """Synthesize `text` to `out_path` (.mp3) with edge-tts. Returns the path."""
    import edge_tts
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    async def _run():
        await edge_tts.Communicate(text, voice,
                                   rate=rate, pitch=pitch, volume=volume).save(out_path)
    asyncio.run(_run())

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError('edge-tts produced no audio '
                           '(check the voice id and your internet connection).')
    return out_path


def main():
    p = argparse.ArgumentParser(
        description='Generate TTS speech audio in a male or female voice.',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('text', help='the text to speak')
    p.add_argument('--gender', '-g', choices=['male', 'female'], default='female',
                   help='voice gender (default: female)')
    p.add_argument('--voice', default=None,
                   help='override with a specific edge-tts voice id')
    p.add_argument('--output', '-o', default=None,
                   help='output .mp3 path (default: output/audio/tts_output.mp3)')
    p.add_argument('--rate', default='-4%',
                   help="speaking rate, e.g. '-10%%', '+0%%' (default: -4%%)")
    p.add_argument('--pitch', default='+0Hz',
                   help="pitch, e.g. '-2Hz' (default: +0Hz)")
    p.add_argument('--volume', default='+0%',
                   help="volume, e.g. '+0%%' (default: +0%%)")
    args = p.parse_args()

    try:
        import edge_tts  # noqa: F401
    except ImportError:
        sys.exit('ERROR: edge-tts is not installed.  Run:  pip install edge-tts')

    if not args.text.strip():
        sys.exit('ERROR: text is empty.')

    voice = args.voice or VOICES[args.gender]
    if args.output:
        out = args.output if args.output.lower().endswith('.mp3') else args.output + '.mp3'
    else:
        os.makedirs(AUDIO_DIR, exist_ok=True)
        out = str(AUDIO_DIR / 'tts_output.mp3')

    print(f'TTS  ->  {out}')
    print(f'  voice : {voice}  ({args.gender})   rate {args.rate}, pitch {args.pitch}')
    print(f'  words : {len(args.text.split())}')
    generate_audio(args.text, out, voice, args.rate, args.pitch, args.volume)

    d = _duration(out)
    print(f'  size  : {os.path.getsize(out) // 1024} KB')
    print(f'  duration : {d:.2f}s   <- use this as S in sequencer.py' if d is not None
          else '  duration : (install ffmpeg to measure)')


if __name__ == '__main__':
    main()
