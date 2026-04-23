#!/usr/bin/env python3
"""Record the D3 timelapse HTML to an MP4 using Playwright + ffmpeg.

Spins up a headless Chromium, loads ``ais_timelapse_d3.html``, waits
for ``window.__timelapse_ready``, resets the animation via
``window.__timelapse_seek(0)``, captures PNG frames at a fixed cadence,
then encodes them to MP4 with libx264.

Usage:
    python scripts/record_timelapse_d3.py                       # defaults
    python scripts/record_timelapse_d3.py --fps 30
    python scripts/record_timelapse_d3.py --duration 15 --out ais_d3.mp4

Requirements:
    pip install playwright
    playwright install chromium
    ffmpeg on PATH
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
        default=str(_REPO_ROOT / "ais_timelapse_d3.html"),
        help="Path to the timelapse HTML to record.",
    )
    p.add_argument(
        "--output",
        default=str(_REPO_ROOT / "ais_timelapse_d3.mp4"),
        help="Output MP4 path.",
    )
    p.add_argument(
        "--width", type=int, default=720, help="Viewport width in CSS px.",
    )
    p.add_argument(
        "--height", type=int, default=900, help="Viewport height in CSS px.",
    )
    p.add_argument(
        "--fps", type=float, default=30.0,
        help="Capture framerate (also the MP4 framerate). Default 30.",
    )
    p.add_argument(
        "--duration", type=float, default=15.0,
        help="Recording duration in seconds. Default 15.",
    )
    p.add_argument(
        "--warmup", type=float, default=1.5,
        help="Seconds to animate after load before capture starts. "
             "Lets the trail buffer develop a baseline before frame 0.",
    )
    p.add_argument(
        "--crf", type=int, default=18,
        help="libx264 CRF quality (lower = better). Default 18.",
    )
    p.add_argument(
        "--keep-frames", action="store_true",
        help="Keep captured PNG frames (default: delete after encode).",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
    )
    return p.parse_args()


async def _capture(
    html_url: str,
    frames_dir: Path,
    width: int,
    height: int,
    fps: float,
    duration: float,
    warmup: float,
    log: logging.Logger,
) -> int:
    # Import here so the script can show --help without Playwright installed.
    try:
        from playwright.async_api import async_playwright
    except ImportError as err:  # noqa: PERF203
        log.error(
            "Playwright is not installed. Run: pip install playwright && "
            "playwright install chromium",
        )
        raise SystemExit(2) from err

    total_frames = int(round(fps * duration))
    frame_interval = 1.0 / fps

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=[
            "--disable-gpu-vsync",
            "--disable-renderer-backgrounding",
            "--disable-background-timer-throttling",
            "--hide-scrollbars",
        ])
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=2,
        )
        page = await context.new_page()

        log.info("Loading %s …", html_url)
        await page.goto(html_url, wait_until="networkidle")

        # Wait for the template to mark itself ready.
        await page.wait_for_function(
            "() => window.__timelapse_ready === true",
            timeout=30_000,
        )
        log.info("Timelapse ready. Warming up for %.1fs …", warmup)
        await asyncio.sleep(warmup)

        # Reset to t=0 for a clean capture.
        await page.evaluate("() => window.__timelapse_seek && window.__timelapse_seek(0)")

        log.info(
            "Capturing %d frames at %.1f fps → %s",
            total_frames, fps, frames_dir,
        )
        loop = asyncio.get_event_loop()
        start = loop.time()
        for i in range(total_frames):
            target = start + i * frame_interval
            now = loop.time()
            if now < target:
                await asyncio.sleep(target - now)
            await page.screenshot(
                path=str(frames_dir / f"frame_{i:05d}.png"),
                type="png",
                full_page=False,
                omit_background=False,
            )

        await context.close()
        await browser.close()
    return total_frames


def _encode_mp4(frames_dir: Path, output: Path, fps: float, crf: int, log: logging.Logger) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        log.error("ffmpeg not found on PATH — install it and re-run.")
        raise SystemExit(3)

    cmd = [
        ffmpeg,
        "-y",
        "-framerate", f"{fps}",
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        # Ensure even dimensions for libx264.
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(output),
    ]
    log.info("Encoding MP4 …")
    log.debug("cmd: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
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

    output = Path(args.output).resolve()
    tmp_root = Path(tempfile.mkdtemp(prefix="neptune_timelapse_"))
    frames_dir = tmp_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Serve the HTML from a tiny HTTP server so file:// is not needed
    # (Chromium blocks module scripts under file://).
    import http.server
    import socketserver
    import threading

    serve_dir = html_path.parent
    served_name = html_path.name

    class _QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_args, **_kwargs) -> None:  # noqa: D401
            return

        def translate_path(self, path):  # type: ignore[override]
            # Map everything under /served/ to the HTML file's directory.
            # All other paths fall through (shouldn't happen in practice).
            return str(serve_dir / path.lstrip("/"))

    with socketserver.TCPServer(("127.0.0.1", 0), _QuietHandler) as server:
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{port}/{served_name}"
        try:
            asyncio.run(_capture(
                url, frames_dir,
                args.width, args.height, args.fps, args.duration, args.warmup,
                log,
            ))
            _encode_mp4(frames_dir, output, args.fps, args.crf, log)
        finally:
            server.shutdown()

    if not args.keep_frames:
        shutil.rmtree(tmp_root, ignore_errors=True)
    else:
        log.info("Kept frames at %s", frames_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
