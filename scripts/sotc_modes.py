#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""sotc: game modes and engine settings for Shadow of the Colossus EU 1.01 (CUSA08809).

    python scripts/sotc_modes.py GAME --out out --modes af16,fps60 --cvars "r_motionBlurEnable=0 ..."

Writes two files for the loader:
  out/patches.bin  code patches (BBPATCH2, bb-probe --patches):
                   fps60  removes the game's 30 FPS cap (illusion's GoldHEN patch, the shadPS4
                          SotC fork's "SotC-Modes")
  out/cmdline.bin  the engine command line the game appends to its own (SOTCCMD1,
                   bb-probe --sotc-cmdline): cvars of the Bluepoint engine, e.g.
                   +r_motionBlurEnable=0. Modes that are only cvars:
                   af16   16x anisotropic filtering on every texture
The original bytes are checked first: another game version gets neither file's changes.
"""
import argparse
import re
from pathlib import Path
import struct
from prepare import parse_self

BASE = 0x400000
# The lea that loads the string the game appends to its command line ("+VSyncWindow=10%").
CMDLINE_LEA = 0x009367b9
CMDLINE_ORIGINAL = '+VSyncWindow=10%'
# Code patches: mode -> [(address, expected original bytes, new bytes)]
MODES = {
    'fps60': [
        # the frame limiter's import stub (sceVideoOutSetFlipRate) returns 0
        (0x01390090, bytes.fromhex('ff25a20567'), bytes.fromhex('31c0c3')),
    ],
    'nobars': [
        # the cutscene letterbox command takes a bar height of 0: neither the UI bars
        # (BPBarsWrapper) nor the original code's two black quads are drawn
        (0x009e645b, bytes.fromhex('c57a2cf8'), bytes.fromhex('4531ff90')),
    ],
}
# aspectWxH (experimental): the display aspect ratio the game stores at start (16/9, a float in
# the display object; mov dword [rbx+4], 0x3fe38e39) becomes W/H.
ASPECT_SITE = 0x00495b3e
ASPECT_ORIGINAL = bytes.fromhex('c74304398ee33f')
ASPECT_MODE = re.compile(r'^aspect(\d+)x(\d+)$')
# displayWxH and renderWxH (experimental, PS4 Pro mode only; use both for 21:9, e.g. 3072x1280
# upscaled to 3440x1440). W must be a multiple of 512: other widths pad the tiled rows and
# streaks appeared on characters and in the sky (3456x1440).
# the display buffers (3840x2160) and the render size (2560x1440) the game picks for a PS4 Pro.
# (address, expected bytes, imm32 offset, 'w' or 'h'): mov r32, imm32 in the display init.
DISPLAY_SITES = [
    (0x00495bfc, bytes.fromhex('41bc000f0000'), 2, 'w'),  # display width 3840
    (0x00495bf1, bytes.fromhex('41bf70080000'), 2, 'h'),  # display height 2160
    (0x004a2ae4, bytes.fromhex('b8000f0000'), 1, 'w'),  # another display-size object: width 3840
    (0x004a2b08, bytes.fromhex('b870080000'), 1, 'h'),  # height 2160
]
RENDER_SITES = [
    (0x00495d5b, bytes.fromhex('41bd000a0000'), 2, 'w'),  # render width 2560
    (0x00495d76, bytes.fromhex('41bfa0050000'), 2, 'h'),  # render height 1440
]
# The render target pool (RTAllocator) of a PS4 Pro, 0x23d10000 bytes for 2560x1440, grows with
# the render size (mov eax, imm32).
RT_POOL_SITE, RT_POOL_ORIGINAL, RT_POOL_PRO = 0x0049b760, bytes.fromhex('b80000d123'), 0x23d10000
# Texel sizes of the PS4 Pro half and full render size (1/1280, 1/720, 1/2560, 1/1440), copied
# into a shader constant block (vmovups at 0x9e3dd8); renderWxH sets them for WxH.
TEXEL_SIZES, TEXEL_SIZES_ORIGINAL = 0x01668b90, struct.pack('<4f', 1 / 1280, 1 / 720, 1 / 2560, 1 / 1440)
# The UI lays out in the display buffer's pixels, from float tables {base PS4, PS4 Pro} of the
# width {1920, 3840} and height {1080, 2160}; displayWxH changes the PS4 Pro entries.
UI_SIZE_TABLES = [(0x013c34a0, 'w'), (0x013c34a8, 'h'), (0x013d7518, 'w'), (0x013d7520, 'h'),
                  (0x01667f14, 'w'), (0x01667f1c, 'h'), (0x01667f24, 'w'), (0x01667f2c, 'h'),
                  (0x01667f34, 'w'), (0x01667f3c, 'h'), (0x01667f54, 'w'), (0x01667f5c, 'h'),
                  (0x01667f84, 'w'), (0x01667f8c, 'h')]
UI_SIZE_ORIGINAL = {'w': struct.pack('<ff', 1920.0, 3840.0), 'h': struct.pack('<ff', 1080.0, 2160.0)}
SIZE_MODE = re.compile(r'^(display|render)(\d+)x(\d+)$')
# dlss: the camera jitter of the main view (vmovlps [rbx+0x80], xmm0 at 0x4bbbb1, after the
# jitter function at 0x4bb950: (2 * Halton(2,3) - 1) * (s / width, s / height)) is also stored
# at image offset DLSS_JITTER, in the padding between the eboot and the first module, where the
# GPU library reads it for DLSS. Two 14-byte padding caves after no-return calls hold the code.
DLSS_JITTER = 0x395f000
DLSS_HOOK, DLSS_CAVE_A, DLSS_CAVE_B = 0x4bbbb1, 0x4bb942, 0x4bc242
JMP, NOP = bytes([0xe9]), bytes([0x90])


def rel32(source_end, target):
    return struct.pack('<i', target - source_end)


DLSS_STORE = bytes.fromhex('c5f8138380000000')  # vmovlps [rbx+0x80], xmm0
MODES['dlss'] = [
    (DLSS_HOOK, DLSS_STORE, JMP + rel32(DLSS_HOOK + 5, DLSS_CAVE_A) + NOP * 3),
    (DLSS_CAVE_A, NOP * 13, DLSS_STORE + JMP + rel32(DLSS_CAVE_A + 13, DLSS_CAVE_B)),
    (DLSS_CAVE_B, NOP * 13,
     bytes.fromhex('c5f81305') + rel32(DLSS_CAVE_B + 8, BASE + DLSS_JITTER)  # vmovlps [rip+d], xmm0
     + JMP + rel32(DLSS_CAVE_B + 13, DLSS_HOOK + 8)),
]
# Modes made of cvars only.
CVAR_MODES = {
    'af16': ['progSamplerAnisoLevel=16', 'progSamplerAnisoLevel_BasePS4=16', 'progSamplerAnisoLevel_Perf=16',
             'progSamplerAnisoLevel_Cinematic=16', 'linearSamplerAnisoLevel=16'],
}
# EU 1.01: the command line lea and the fps60 site.
VERSION_CHECKS = [(CMDLINE_LEA, bytes.fromhex('488d352debad00')), (0x01390090, bytes.fromhex('ff25a20567'))]
CVAR = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*(=[-+.0-9A-Za-z_%]+)?$')  # NAME=VALUE or a flag such as nointro


def image_bytes(elf, ph, vaddr, length):
    for p in ph:
        if p['type'] in (1, 0x61000010) and p['vaddr'] <= vaddr and vaddr + length <= p['vaddr'] + p['filesz']:
            offset = p['offset'] + vaddr - p['vaddr']
            return bytes(elf[offset:offset + length])
    return None


def is_supported(elf, ph):
    return all(image_bytes(elf, ph, address - BASE, len(original)) == original for address, original in VERSION_CHECKS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('game', type=Path)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent.parent / 'out')
    parser.add_argument('--modes', default='', help='comma-separated: ' + ', '.join([*MODES, *CVAR_MODES]))
    parser.add_argument('--cvars', default='', help='engine cvars NAME=VALUE separated by spaces')
    args = parser.parse_args()
    elf, _, ph, _, _ = parse_self((args.game / 'eboot.bin').read_bytes())
    supported = is_supported(elf, ph)
    writes, cvars = [], []
    for mode in filter(None, (m.strip().lower() for m in args.modes.split(','))):
        aspect = ASPECT_MODE.match(mode)
        if aspect:
            if image_bytes(elf, ph, ASPECT_SITE - BASE, len(ASPECT_ORIGINAL)) != ASPECT_ORIGINAL:
                print(f'Mode {mode}: aspect site not found, not applied')
                continue
            ratio = int(aspect.group(1)) / int(aspect.group(2))
            writes.append((ASPECT_SITE + 3 - BASE, struct.pack('<f', ratio)))
            print(f'Mode {mode}: display aspect {ratio:.4f}')
            continue
        sized = SIZE_MODE.match(mode)
        if sized:
            kind, size = sized.group(1), {'w': int(sized.group(2)), 'h': int(sized.group(3))}
            sites = DISPLAY_SITES if kind == 'display' else RENDER_SITES
            if any(image_bytes(elf, ph, a - BASE, len(o)) != o for a, o, _, _ in sites):
                print(f'Mode {mode}: {kind} size sites not found, not applied')
                continue
            writes += [(a + i - BASE, struct.pack('<I', size[k])) for a, _, i, k in sites]
            if kind == 'display':
                for a, k in UI_SIZE_TABLES:
                    if image_bytes(elf, ph, a - BASE, 8) == UI_SIZE_ORIGINAL[k]:
                        writes.append((a + 4 - BASE, struct.pack('<f', float(size[k]))))
            elif image_bytes(elf, ph, TEXEL_SIZES - BASE, 16) == TEXEL_SIZES_ORIGINAL:
                w, h = size['w'], size['h']
                writes.append((TEXEL_SIZES - BASE, struct.pack('<4f', 2 / w, 2 / h, 1 / w, 1 / h)))
            if kind == 'render' and image_bytes(elf, ph, RT_POOL_SITE - BASE, len(RT_POOL_ORIGINAL)) == RT_POOL_ORIGINAL:
                # Scaled by the pixel count, with 25% headroom, in 64 KiB units.
                scale = max(1.0, size['w'] * size['h'] / (2560 * 1440)) * 1.25
                pool = (int(RT_POOL_PRO * scale) + 0xffff) & ~0xffff
                writes.append((RT_POOL_SITE + 1 - BASE, struct.pack('<I', pool)))
                print(f'Mode {mode}: render target pool {pool / 2**20:.0f} MiB')
            print(f'Mode {mode}: PS4 Pro {kind} size {size["w"]}x{size["h"]}')
            continue
        if mode not in MODES and mode not in CVAR_MODES:
            raise SystemExit(f'unknown mode {mode!r} (known: {", ".join([*MODES, *CVAR_MODES])})')
        if not supported:
            print(f'Mode {mode}: this eboot is not EU 1.01, not applied')
            continue
        writes += [(address - BASE, data) for address, _, data in MODES.get(mode, [])]
        cvars += CVAR_MODES.get(mode, [])
        print(f'Mode {mode}: on')
    for item in args.cvars.split():
        item = item.lstrip('+')
        if not CVAR.match(item):
            raise SystemExit(f'bad cvar {item!r} (NAME=VALUE or NAME)')
        cvars.append(item)
    blob = struct.pack('<8sQQ', b'BBPATCH2', BASE, len(writes))
    for offset, data in writes:
        blob += struct.pack('<QQ', offset, len(data)) + data
    (args.out / 'patches.bin').write_bytes(blob)
    cmdline = args.out / 'cmdline.bin'
    if supported and cvars:
        text = ' '.join([CMDLINE_ORIGINAL, *('+' + c for c in cvars)])
        cmdline.write_bytes(struct.pack('<8sQ', b'SOTCCMD1', CMDLINE_LEA - BASE) + text.encode('ascii'))
    elif cmdline.exists():
        cmdline.unlink()


if __name__ == '__main__':
    main()
