#!/usr/bin/env python3
"""Milestone 2 - sequence an idle -> speaking -> idle clip (no audio).

Takes the same boomerang/ease logic as pipeline.ipynb and exposes it as a CLI.

Usage
-----
    python sequencer.py  IDLE_OR_HEADSHOT  SPEAKING_ANIMATION  S  [options]

Positional arguments
    IDLE_OR_HEADSHOT   an IMAGE  -> used as a static headshot (a 3s freeze is
                       placed at the start and end), or
                       a VIDEO   -> used as an idle animation (eased boomerang):
                         * duration > 1.5s : first 1.5s + reverse (single boomerang)
                         * duration < 1.5s : loop (clip + reverse) to fill 3s
    SPEAKING_ANIMATION a VIDEO turned into an eased boomerang of EXACTLY S seconds
    S                  speaking-segment length, in seconds (float)

The result timeline is:  idle(IDLE_DURATION) + speaking(S) + idle(IDLE_DURATION).

Requires ffmpeg + ffprobe on PATH.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}

# Default output location: <this script's folder>/output/sequenced/
SEQ_DIR = Path(__file__).resolve().parent / 'output' / 'sequenced'


# ── ffprobe / ffmpeg helpers ─────────────────────────────────
def _dur(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                          '-of', 'json', str(path)], capture_output=True, text=True).stdout
    return float(json.loads(out)['format']['duration'])

def _fps(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                          '-show_entries', 'stream=r_frame_rate', '-of', 'json',
                          str(path)], capture_output=True, text=True).stdout
    n, d = json.loads(out)['streams'][0]['r_frame_rate'].split('/')
    return float(n) / float(d)

def _res(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                          '-show_entries', 'stream=width,height', '-of', 'json',
                          str(path)], capture_output=True, text=True).stdout
    s = json.loads(out)['streams'][0]
    return int(s['width']), int(s['height'])

def _ff(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('ffmpeg failed:\n' + (r.stderr or '')[-1500:])


# ── Boomerang building blocks (identical logic to the pipeline) ──
def _eased_forward(src, out, half, fps, e_src, e_out):
    """First `half` seconds of src, with the last e_src s slowed to e_out s."""
    half = float(half)
    normal = half - e_src
    if normal > 0 and e_src > 0 and e_out > 0:
        factor = e_out / e_src                      # >1 = slower
        fc = (f'[0:v]trim=0:{normal:.6f},setpts=PTS-STARTPTS[a];'
              f'[0:v]trim={normal:.6f}:{half:.6f},setpts={factor:.6f}*(PTS-STARTPTS)[b];'
              f'[a][b]concat=n=2:v=1,fps={fps:.6f}[v]')
    else:                                            # too short to ease - plain clip
        fc = f'[0:v]trim=0:{half:.6f},setpts=PTS-STARTPTS,fps={fps:.6f}[v]'
    _ff(['ffmpeg', '-y', '-i', str(src), '-filter_complex', fc, '-map', '[v]', '-an',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', str(out)])
    return out


def _boomerang(fwd, out, fps, drop_first=False):
    """forward + reverse(forward), minus the duplicate turnaround frame.
    drop_first also removes the very first frame (for a seamless join after a
    previous piece that already ends on this same frame)."""
    base = ('[0:v]split[x][y];'
            '[y]reverse,trim=start_frame=1,setpts=PTS-STARTPTS[r];'
            f'[x][r]concat=n=2:v=1,fps={fps:.6f}')
    fc = (base + f',trim=start_frame=1,setpts=PTS-STARTPTS,fps={fps:.6f}[v]') if drop_first \
         else (base + '[v]')
    _ff(['ffmpeg', '-y', '-i', str(fwd), '-filter_complex', fc, '-map', '[v]', '-an',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', str(out)])
    return out


def build_looped_boomerang(src, out, target, half_window, e_src, e_out):
    """Eased boomerang clip of EXACTLY `target` seconds.

    `half_window` is the forward half-length of one seamless unit:
      * idle  -> half_window = target/2 (e.g. 1.5s for a 3s idle); a source
        longer than that yields a single 1.5s + reverse boomerang.
      * speaking -> half_window = V/2 (half the talking clip), tiled to reach S.
    A source shorter than half_window is looped (unit + reverse) to fill target.
    (This is the generalized form of the pipeline's build_speaking().)
    """
    src, out = str(src), str(out)
    if not os.path.exists(src):
        raise FileNotFoundError(f'Input not found: {src}')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    V, fps = _dur(src), _fps(src)
    tmp = tempfile.mkdtemp()
    posix = lambda p: str(p).replace('\\', '/')

    hw = min(float(half_window), V)     # unit half-window (can't exceed the source)
    L = (2.0 * hw + 2.0 * (e_out - e_src) - 1.0 / fps) if hw > e_src else (2.0 * hw - 1.0 / fps)
    N = max(0, int(target // L)) if L > 0 else 0
    R = target - N * L

    pieces = []
    if N >= 1:
        fwd_u = _eased_forward(src, f'{tmp}/fwd_u.mp4', hw, fps, e_src, e_out)
        unit_full = _boomerang(fwd_u, f'{tmp}/unit_full.mp4', fps, drop_first=False)
        pieces.append(unit_full)
        if N >= 2:
            unit_nodup = _boomerang(fwd_u, f'{tmp}/unit_nodup.mp4', fps, drop_first=True)
            pieces += [unit_nodup] * (N - 1)
        if R > 0.3:                                  # leftover worth its own boomerang
            fwd_r = _eased_forward(src, f'{tmp}/fwd_r.mp4', R / 2.0, fps, e_src, e_out)
            pieces.append(_boomerang(fwd_r, f'{tmp}/rem.mp4', fps, drop_first=True))
    else:                                            # target <= one unit: single boomerang
        R = target
        fwd_r = _eased_forward(src, f'{tmp}/fwd_r.mp4', target / 2.0, fps, e_src, e_out)
        pieces.append(_boomerang(fwd_r, f'{tmp}/rem.mp4', fps, drop_first=False))

    listf = f'{tmp}/list.txt'
    with open(listf, 'w') as f:
        for p in pieces:
            f.write(f"file '{posix(p)}'\n")
    concat_tmp = f'{tmp}/concat.mp4'
    _ff(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', listf, '-an',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-r', f'{fps:.6f}', concat_tmp])

    # Stretch the assembly to land on exactly `target` (absorbs dropped frames).
    D = _dur(concat_tmp)
    factor = target / D
    fc = f'[0:v]setpts={factor:.6f}*PTS,fps={fps:.6f}[v]'
    _ff(['ffmpeg', '-y', '-i', concat_tmp, '-filter_complex', fc, '-map', '[v]', '-an',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', out])

    print(f'  V={V:.2f}s | half={hw:.2f}s | unit~{L:.2f}s | target={target:.2f}s  ->  '
          f'{N} unit(s) + {R:.2f}s remainder  ->  {_dur(out):.2f}s @ {fps:.2f} fps')
    return out


def make_static_video(image, out, duration, W, H, fps):
    """A `duration`-second freeze of a still image at WxH/fps (aspect-preserved)."""
    vf = (f'scale={W}:{H}:force_original_aspect_ratio=decrease,'
          f'pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p')
    _ff(['ffmpeg', '-y', '-loop', '1', '-i', str(image), '-t', f'{duration:.3f}',
         '-vf', vf, '-r', f'{fps:.6f}',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', str(out)])
    print(f'  static freeze -> {duration:.2f}s @ {W}x{H} {fps:.2f} fps')
    return out


def build_sequence(idle_clip, speaking_clip, out, idle_dur, crossfade):
    """Stitch idle(idle_dur) + speaking(S) + idle(idle_dur). (Pipeline logic.)"""
    for f in (idle_clip, speaking_clip):
        if not os.path.exists(f):
            raise FileNotFoundError(f'Input not found: {f}')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fps  = _fps(speaking_clip)
    W, H = _res(speaking_clip)
    tmp  = tempfile.mkdtemp()

    k = idle_dur / _dur(idle_clip)
    idle_seg = f'{tmp}/idle_seg.mp4'
    _ff(['ffmpeg', '-y', '-i', str(idle_clip), '-filter_complex',
         f'[0:v]setpts={k:.6f}*PTS,fps={fps:.6f},scale={W}:{H},setsar=1,format=yuv420p[v]',
         '-map', '[v]', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', idle_seg])

    speak_seg = f'{tmp}/speak_seg.mp4'
    _ff(['ffmpeg', '-y', '-i', str(speaking_clip), '-filter_complex',
         f'[0:v]fps={fps:.6f},scale={W}:{H},setsar=1,format=yuv420p[v]',
         '-map', '[v]', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', speak_seg])
    S = _dur(speak_seg)

    if crossfade and crossfade > 0:
        cf = float(crossfade)
        off1 = idle_dur - cf
        off2 = idle_dur + S - 2 * cf
        fc = (f'[0:v]settb=AVTB[a];[1:v]settb=AVTB[b];[2:v]settb=AVTB[c];'
              f'[a][b]xfade=transition=fade:duration={cf:.6f}:offset={off1:.6f}[ab];'
              f'[ab][c]xfade=transition=fade:duration={cf:.6f}:offset={off2:.6f}[v]')
        mode = f'crossfade {cf:.2f}s'
    else:
        fc = '[0:v][1:v][2:v]concat=n=3:v=1[v]'
        mode = 'hard cut'

    _ff(['ffmpeg', '-y', '-i', idle_seg, '-i', speak_seg, '-i', idle_seg,
         '-filter_complex', fc, '-map', '[v]', '-an',
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-r', f'{fps:.6f}', str(out)])

    total = _dur(out)
    print(f'  timeline ({mode}):  0.00 -> {idle_dur:.2f} idle | '
          f'{idle_dur:.2f} -> {idle_dur+S:.2f} speaking (S={S:.2f}s) | '
          f'{idle_dur+S:.2f} -> {2*idle_dur+S:.2f} idle')
    print(f'  done -> {out}   ({total:.2f}s, {W}x{H} @ {fps:.2f} fps, no audio)')
    return out


def is_image(path):
    return Path(path).suffix.lower() in IMAGE_EXTS


def main():
    p = argparse.ArgumentParser(
        description='Sequence idle -> speaking -> idle (no audio).',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('idle', metavar='IDLE_OR_HEADSHOT',
                   help='image (static headshot) or video (idle animation)')
    p.add_argument('speaking', metavar='SPEAKING_ANIMATION', help='talking-head video')
    p.add_argument('S', type=float, help='speaking-segment length in seconds')
    p.add_argument('--output', '-o', default=None, help='output mp4 path')
    p.add_argument('--idle-duration', type=float, default=3.0,
                   help='idle length at start and end (default 3.0)')
    p.add_argument('--ease-src', type=float, default=0.2,
                   help='source seconds slowed at each turnaround (default 0.2)')
    p.add_argument('--ease-out', type=float, default=0.3,
                   help='stretched length of the eased tail (default 0.3)')
    p.add_argument('--crossfade', type=float, default=0.0,
                   help='crossfade seconds at the idle<->speaking joins (default 0 = hard cut)')
    args = p.parse_args()

    if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:
        sys.exit('ERROR: ffmpeg and ffprobe must be installed and on PATH.')
    for f in (args.idle, args.speaking):
        if not os.path.exists(f):
            sys.exit(f'ERROR: input not found: {f}')
    if args.S <= 0:
        sys.exit('ERROR: S must be > 0.')

    if args.output:
        out = args.output
    else:
        os.makedirs(SEQ_DIR, exist_ok=True)
        out = str(SEQ_DIR / (Path(args.speaking).stem + '_sequenced.mp4'))
    idle_dur = args.idle_duration
    tmp = tempfile.mkdtemp()

    # 1) Speaking clip -> exactly S (half-video eased boomerang, tiled).
    print(f'[1/3] speaking -> {args.S:.2f}s  ({Path(args.speaking).name})')
    speak_clip = os.path.join(tmp, 'speaking.mp4')
    build_looped_boomerang(args.speaking, speak_clip, args.S,
                           _dur(args.speaking) / 2.0, args.ease_src, args.ease_out)
    W, H = _res(speak_clip)
    fps = _fps(speak_clip)

    # 2) Idle clip -> exactly idle_dur (static freeze, or eased/looped boomerang).
    if is_image(args.idle):
        print(f'[2/3] static headshot -> {idle_dur:.2f}s freeze  ({Path(args.idle).name})')
        idle_clip = os.path.join(tmp, 'idle.mp4')
        make_static_video(args.idle, idle_clip, idle_dur, W, H, fps)
    else:
        print(f'[2/3] idle animation -> {idle_dur:.2f}s boomerang  ({Path(args.idle).name})')
        idle_clip = os.path.join(tmp, 'idle.mp4')
        build_looped_boomerang(args.idle, idle_clip, idle_dur,
                               idle_dur / 2.0, args.ease_src, args.ease_out)

    # 3) Stitch idle + speaking + idle.
    print(f'[3/3] stitch -> {out}')
    build_sequence(idle_clip, speak_clip, out, idle_dur, args.crossfade)


if __name__ == '__main__':
    main()
