"""Read the committed release pin the documentation build renders results from.

The lock is a required build input rather than a fallback. It either names one
benchmark version of a Distribution Release repository, or it declares in as
many words that no release is pinned. Results updates are routine revisions of
a version, so a build reads the repository's latest revision and refuses one
that names another version; a new version is a deliberate edit of this file. A build that cannot read the lock stops instead of publishing pages
that quietly name no release.

The lock names no task. A release declares the tasks it released in its own
manifest, and the site renders a Leaderboard and an Explore for exactly those.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# The committed lock's file name, beside the authored library documentation.
LOCK_NAME = "benchmark.lock"

# The value declaring, in as many words, that this build pins no release.
NO_RELEASE = "none"

# Every setting one pinned-release lock states.
PIN_SETTINGS = ("repo_id", "version")


@dataclass(frozen=True)
class BenchmarkPin:
    """The benchmark version a build renders results from.

    Attributes:
        repo_id: The dataset repository the release is published in.
        version: The benchmark version the build renders; the release at the
            repository's latest revision must name it.
    """

    repo_id: str
    version: str


def read_benchmark_lock(path: Path) -> BenchmarkPin | None:
    """Read the release lock at *path*.

    Args:
        path: The committed lock file.

    Returns:
        The pinned release, or ``None`` where the lock declares that this
        build pins none.

    Raises:
        ValueError: The lock is absent, is not made of ``key = value``
            settings, or states a pin this build cannot act on.
    """
    if not path.is_file():
        raise ValueError(
            f"The documentation build requires its release lock at {path}."
        )
    settings = _read_settings(path)
    unknown = sorted(set(settings) - {"release", *PIN_SETTINGS})
    if unknown:
        raise ValueError(f"{path} states unknown setting(s): {unknown}.")
    if "release" in settings:
        if settings != {"release": NO_RELEASE}:
            raise ValueError(
                f"{path} declares release {settings['release']!r}: a lock either "
                f"states 'release = {NO_RELEASE}' alone or names a pinned release."
            )
        return None
    missing = [name for name in PIN_SETTINGS if name not in settings]
    if missing:
        raise ValueError(f"{path} is missing the setting(s): {missing}.")
    empty = [name for name in PIN_SETTINGS if not settings[name]]
    if empty:
        raise ValueError(f"{path} states the setting(s) {empty} with no value.")
    return BenchmarkPin(repo_id=settings["repo_id"], version=settings["version"])


def _read_settings(path: Path) -> dict[str, str]:
    """Return the lock's ``key = value`` settings, ignoring blanks and comments."""
    settings: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        key, separator, value = entry.partition("=")
        if not separator:
            raise ValueError(f"{path} line {entry!r} is not a 'key = value' setting.")
        key = key.strip()
        if key in settings:
            raise ValueError(f"{path} states the setting {key!r} twice.")
        settings[key] = value.strip()
    return settings
