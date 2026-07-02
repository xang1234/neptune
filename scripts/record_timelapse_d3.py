#!/usr/bin/env python3
"""Record the D3 timelapse HTML to an MP4 with deterministic frame stepping.

Unlike real-time screen capture, this drives the animation clock
manually: ``window.__tl_render_frame(i)`` advances the timelapse by
exactly ``1/fps`` per captured frame, so the encoded video is perfectly
smooth regardless of how long each screenshot takes.

Frames are captured at 2× device scale and downscaled by ffmpeg for
crisp supersampled output.

Usage:
    python scripts/record_timelapse_d3.py --html ais_timelapse_d3_corridors.html
    python scripts/record_timelapse_d3.py --duration 15 --warmup 4

Requirements: pip install playwright && playwright install chromium; ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--html",
        default=str(_REPO_ROOT / "ais_timelapse_d3_corridors.html"),
        help="Path to the timelapse HTML to record.",
    )
    p.add_argument("--output", default=None, help="Output MP4 path.")
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--fps", type=int, default=60, help="Output framerate. Default 60.")
    p.add_argument(
        "--duration", type=float, default=15.0,
        help="Captured video duration in seconds. Default 15.",
    )
    p.add_argument(
        "--warmup", type=float, default=0.0,
        help="Seconds of animation rendered (not captured) before frame 0, "
             "so trails already exist at the start of the video.",
    )
    p.add_argument("--crf", type=int, default=17, help="libx264 CRF. Default 17.")
    p.add_argument("--keep-frames", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


async def _capture(
    html_url: str,
    frames_dir: Path,
    args: argparse.Namespace,
    log: logging.Logger,
) -> None:
    try:
        from playwright.async_api import async_playwright
    except ImportError as err:
        log.error(
            "Playwright is not installed. Run: pip install playwright && "
            "playwright install chromium",
        )
        raise SystemExit(2) from err

    warm_frames = int(round(args.warmup * args.fps))
    total_frames = warm_frames + int(round(args.duration * args.fps))

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=[
            "--disable-lcd-text",
            "--hide-scrollbars",
        ])
        context = await browser.new_context(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=2,
        )
        page = await context.new_page()

        log.info("Loading %s …", html_url)
        await page.goto(html_url, wait_until="networkidle")
        await page.wait_for_function(
            "() => window.__tl_ready === true", timeout=30_000,
        )
        await page.evaluate("() => window.__tl_set_manual(true)")

        log.info(
            "Rendering %d frames (%d warmup + %d captured) at %d fps …",
            total_frames, warm_frames, total_frames - warm_frames, args.fps,
        )
        for i in range(total_frames):
            await page.evaluate(f"() => window.__tl_render_frame({i})")
            if i < warm_frames:
                continue
            await page.screenshot(
                path=str(frames_dir / f"frame_{i - warm_frames:05d}.png"),
                type="png",
            )
            if (i - warm_frames) % (args.fps * 2) == 0:
                log.info("  frame %d / %d", i - warm_frames, total_frames - warm_frames)

        await context.close()
        await browser.close()


def _encode_mp4(
    frames_dir: Path, output: Path, args: argparse.Namespace, log: logging.Logger,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        log.error("ffmpeg not found on PATH — install it and re-run.")
        raise SystemExit(3)

    cmd = [
        ffmpeg, "-y",
        "-framerate", str(args.fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-vf", f"scale={args.width}:{args.height}:flags=lanczos",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", str(args.crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output),
    ]
    log.info("Encoding MP4 …")
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    log.info("Wrote %s (%.1f MB)", output, output.stat().st_size / 1024 / 1024)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("record")

    html_path = Path(args.html).resolve()
    if not html_path.exists():
        log.error("HTML file not found: %s", html_path)
        log.error("Run scripts/generate_ais_timelapse_d3.py first.")
        return 1
    output = Path(
        args.output or html_path.with_suffix(".mp4")
    ).resolve()

    tmp_root = Path(tempfile.mkdtemp(prefix="neptune_timelapse_"))
    frames_dir = tmp_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Serve over localhost HTTP — Chromium is stricter about file://.
    import http.server
    import socketserver
    import threading

    serve_dir = html_path.parent

    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_args, **_kwargs) -> None:
            return

        def translate_path(self, path):  # type: ignore[override]
            return str(serve_dir / path.lstrip("/"))

    with socketserver.TCPServer(("127.0.0.1", 0), _QuietHandler) as server:
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{port}/{html_path.name}"
        try:
            asyncio.run(_capture(url, frames_dir, args, log))
            _encode_mp4(frames_dir, output, args, log)
        finally:
            server.shutdown()

    if not args.keep_frames:
        shutil.rmtree(tmp_root, ignore_errors=True)
    else:
        log.info("Kept frames at %s", frames_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
