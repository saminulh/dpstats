"""Serve a built site while syncing presentation assets straight from source."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

SITE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = SITE_ROOT.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from mkdocs.livereload import LiveReloadServer


SOURCE_ROOT = SITE_ROOT / "docs"
SITE_CONFIG = SITE_ROOT / "mkdocs.yml"
DEFAULT_OUTPUT = SITE_ROOT.parents[1] / "site"
ASSET_DIRECTORIES = ("javascripts", "stylesheets")
ASSET_URL = re.compile(rb'((?:javascripts|stylesheets)/[^"?#]+\.(?:js|css))"')


class PreviewServer(LiveReloadServer):
    """Live reload with browser caching disabled for source-owned assets."""

    # A page opens a dozen script and style connections at once; the standard
    # listen backlog of five resets the surplus.
    request_queue_size = 128

    def serve_request(self, environ, start_response):  # type: ignore[no-untyped-def]
        def no_cache(status, headers, exc_info=None):  # type: ignore[no-untyped-def]
            headers.extend(
                [
                    ("Cache-Control", "no-store, no-cache, must-revalidate"),
                    ("Pragma", "no-cache"),
                ]
            )
            return start_response(status, headers, exc_info)

        return super().serve_request(environ, no_cache)

    def _inject_js_into_html(self, content, epoch):  # type: ignore[no-untyped-def]
        version = f"?v={epoch}".encode()
        content = ASSET_URL.sub(lambda match: match.group(1) + version + b'"', content)
        return super()._inject_js_into_html(content, epoch)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve an existing MkDocs build and live-reload CSS/JavaScript "
            "without rebuilding generated result pages."
        )
    )
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--build",
        action="store_true",
        help="run one strict build using the configured release or prepared review before serving",
    )
    parser.add_argument("--build-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _build(output: Path) -> None:
    from mkdocs.commands.build import build
    from mkdocs.config import load_config

    build(load_config(config_file=str(SITE_CONFIG), site_dir=str(output), strict=True))


def _build_separately(output: Path) -> None:
    """Build in a child so pandas and MkDocs memory dies before serving."""
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--build-worker",
            "--site-dir",
            str(output),
        ],
        check=True,
        cwd=REPOSITORY_ROOT,
    )


def _sync_assets(output: Path) -> None:
    copied = 0
    for directory in ASSET_DIRECTORIES:
        source = SOURCE_ROOT / directory
        target = output / directory
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            destination = target / path.relative_to(source)
            source_stat = path.stat()
            current = destination.stat() if destination.exists() else None
            if current and (
                current.st_size,
                current.st_mtime_ns,
            ) == (source_stat.st_size, source_stat.st_mtime_ns):
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied += 1
    if copied:
        print(f"Synced {copied} presentation asset(s).")


def main() -> None:
    args = _arguments()
    output = args.site_dir.resolve()
    if args.build_worker:
        _build(output)
        return
    if args.build:
        _build_separately(output)
    if not (output / "index.html").is_file():
        raise SystemExit(
            f"{output} is not a built site; rerun this command with --build."
        )

    def sync() -> None:
        _sync_assets(output)

    sync()
    server = PreviewServer(sync, args.host, args.port, str(output))
    for directory in ASSET_DIRECTORIES:
        server.watch(str(SOURCE_ROOT / directory))
    try:
        server.serve()
    except KeyboardInterrupt:
        server.shutdown(wait=True)


if __name__ == "__main__":
    main()
