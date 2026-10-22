"""Fetch one Distribution Release into a workspace, as a plain download."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import import_module
from pathlib import Path, PurePosixPath
import os
import shutil
import tempfile
from typing import Any, Literal

import huggingface_hub
from tqdm.auto import tqdm

from dpstats.transport.validators import (
    is_immutable_revision,
    is_normalized_relative_path,
)
from dpstats.workspace.layout import (
    RELEASE_MANIFEST_NAME,
    init_workspace,
    read_release_identity,
    workspace_release_root,
)

# The Hugging Face dataset repository the published Distribution Release
# lives in. A library constant rather than a caller-supplied argument: the
# platform publishes exactly one such repository.
DISTRIBUTION_RELEASE_REPOSITORY = "dpstatsbench/distribution"

# Where a download into a directory keeps the hub's own bookkeeping, beneath
# that directory; it is removed before a fetched tree is installed, so no
# released path may use it.
HUB_DOWNLOAD_METADATA_DIRECTORY = ".cache"

# Where `fetch_benchmark` records the immutable commit an installed
# ``official-release/`` was fetched at. No released path may lie beneath
# ``HUB_DOWNLOAD_METADATA_DIRECTORY``, so the record never collides with a
# released file.
INSTALLED_COMMIT_RECORD = f"{HUB_DOWNLOAD_METADATA_DIRECTORY}/commit"

# The revision `fetch_benchmark` installs when the caller names none: the
# repository's default branch, which holds the latest published release. The
# installed manifest records which version that resolved to; nothing here pins.
LATEST_RELEASE_REVISION = "main"


def _released_files(listing: Iterable[Any]) -> list[Any]:
    """Return the released files one recursive repository listing names.

    ``.gitattributes`` belongs to the repository rather than to the release.
    Every other file is joined onto the install root, so one whose path is not
    exactly normalized and relative names a file outside the tree it claims,
    and one beneath the hub's download bookkeeping would be removed with it.

    Raises:
        ValueError: If a listed path is not an installable relative path.
    """
    released: list[Any] = []
    for entry in listing:
        relative = entry.path
        if not hasattr(entry, "size") or relative == ".gitattributes":
            continue
        if (
            not is_normalized_relative_path(relative)
            or PurePosixPath(relative).parts[0] == HUB_DOWNLOAD_METADATA_DIRECTORY
        ):
            raise ValueError(
                f"Distribution Release repository names a path a fetch cannot "
                f"install: {relative!r}."
            )
        released.append(entry)
    return sorted(released, key=lambda entry: entry.path)


def _install_xet_files(
    stage: Path,
    files: Sequence[Any],
    repository_id: str,
    revision: str,
    token: str | Literal[False],
) -> None:
    """Download every Xet-stored file in one transfer group under one access token.

    A per-file download asks the Hub API for its own Xet token, and a release
    of many small files then reaches the account's API rate limit.
    """
    if not files:
        return
    # hf_xet ships no type information.
    xet: Any = import_module("hf_xet")
    metadata = huggingface_hub.get_hf_file_metadata(
        huggingface_hub.hf_hub_url(
            repository_id, files[0].path, repo_type="dataset", revision=revision
        ),
        token=token,
    )
    if metadata.xet_file_data is None:
        raise ValueError(
            f"{files[0].path} is listed as Xet-stored but the Hub served no Xet "
            "data for it."
        )
    headers = {"authorization": f"Bearer {token}"} if token else {}
    with tqdm(
        total=sum(entry.size for entry in files),
        desc="Release files",
        unit="B",
        unit_scale=True,
    ) as progress:

        def report(group: Any, _items: Any) -> None:
            progress.update(group.total_bytes_completed - progress.n)

        with xet.XetSession().new_file_download_group(
            token_refresh_url=metadata.xet_file_data.refresh_route,
            token_refresh_headers=headers,
            progress_callback=report,
        ) as group:
            for entry in files:
                target = stage / entry.path
                target.parent.mkdir(parents=True, exist_ok=True)
                group.start_download_file(
                    xet.XetFileInfo(entry.xet_hash, entry.size),
                    str(target.absolute()),
                )


def _install_plain_files(
    stage: Path,
    files: Sequence[Any],
    repository_id: str,
    revision: str,
    token: str | Literal[False],
) -> None:
    """Download the files stored outside Xet, at most sixteen at a time.

    A download into a directory keeps no second copy in a download cache;
    the hub's bookkeeping for it lands beneath the stage's
    ``HUB_DOWNLOAD_METADATA_DIRECTORY``.
    """
    with ThreadPoolExecutor(max_workers=16) as workers:
        pending = [
            workers.submit(
                huggingface_hub.hf_hub_download,
                repo_id=repository_id,
                repo_type="dataset",
                revision=revision,
                filename=entry.path,
                local_dir=stage,
                token=token,
            )
            for entry in files
        ]
        try:
            for finished in as_completed(pending):
                finished.result()
        finally:
            # Stop queued transfers on failure; the context joins running ones
            # before the caller removes its unfinished staging directory.
            for future in pending:
                future.cancel()


def _fetch_release_tree(
    repository_id: str,
    *,
    revision: str,
    destination: Path | str,
    token: str | Literal[False] = False,
) -> str:
    """Anonymously fetch one whole Distribution Release as a fresh tree.

    Private installer: downloads one immutable revision into a *fresh*
    ``destination`` (refused if anything already exists there). The public
    ``fetch_benchmark`` wraps this with workspace semantics (idempotent
    install, revision replacement, the library's own repository constant and
    ``HF_TOKEN``); a bare-bundle consumer that installs its own destination
    directly (the docs site's build) calls this instead.

    Resolve the requested tag or revision to an immutable commit and install
    the files that commit holds into a fresh ``destination``, apart from the
    repository's own ``.gitattributes``. The commit is the integrity claim: it
    binds every byte the release ships, so a fetch is a download and checks
    nothing further. Each file downloads straight into the staged tree, so no
    second copy stays in a download cache. The installed tree is therefore
    the tree a local cut of the same release holds. It is read-only: its
    Results are resolved and a user's own runs belong in the user's own
    collection. Fetch resolves no policy, creates no Instance, and runs no
    collection step.

    Fetching is anonymous by default and then requires a publicly readable
    repository. Passing ``token`` authenticates every hub call instead, which
    admits a Distribution Release that is not yet publicly readable. Ambient
    Hugging Face credentials are never used.

    Args:
        repository_id: The Hugging Face dataset repository holding the
            Distribution Release, as ``"<owner>/<dataset>"``.
        revision: The tag or commit to resolve to an immutable revision.
        destination: The fresh directory to create with the installed release;
            refused if anything already exists there.
        token: An access token authenticating every hub call, or ``False``
            for an anonymous fetch.

    Returns:
        The immutable commit the requested revision resolved to, whose files
        ``destination`` now holds.

    Raises:
        ValueError: If ``token`` is neither a nonempty string nor ``False``,
            the destination already exists, the resolved revision is not
            immutable, an anonymous fetch reaches a private repository, the
            repository names a path a fetch cannot install, or the Hub serves
            no Xet data for a file it lists as Xet-stored.

    Example:
        >>> commit = _fetch_release_tree(  # doctest: +SKIP
        ...     "<owner>/<dataset>",
        ...     revision="<tag-or-commit>",
        ...     destination="benchmark-release",
        ... )
    """
    if not (isinstance(token, str) and token) and token is not False:
        raise ValueError(
            "Benchmark fetch takes a nonempty token string or token=False; "
            "ambient Hugging Face credentials are never used."
        )
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Benchmark fetch destination already exists: {destination}.")
    api = huggingface_hub.HfApi(token=token)
    info = api.repo_info(repo_id=repository_id, repo_type="dataset", revision=revision)
    if token is False and getattr(info, "private", None) is not False:
        raise ValueError("Benchmark fetch requires an anonymously readable repository.")
    pinned_revision = getattr(info, "sha", None)
    if not is_immutable_revision(pinned_revision):
        raise ValueError("Benchmark source did not resolve to an immutable revision.")
    released = _released_files(
        api.list_repo_tree(
            repo_id=repository_id,
            repo_type="dataset",
            revision=pinned_revision,
            recursive=True,
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        stage = Path(temporary) / "benchmark"
        stage.mkdir()
        _install_xet_files(
            stage,
            [entry for entry in released if entry.xet_hash],
            repository_id,
            pinned_revision,
            token,
        )
        _install_plain_files(
            stage,
            [entry for entry in released if not entry.xet_hash],
            repository_id,
            pinned_revision,
            token,
        )
        shutil.rmtree(stage / HUB_DOWNLOAD_METADATA_DIRECTORY, ignore_errors=True)
        if destination.exists() or destination.is_symlink():
            raise ValueError(
                f"Benchmark fetch destination already exists: {destination}."
            )
        stage.rename(destination)
    return pinned_revision


def fetch_benchmark(
    workspace: Path | str,
    *,
    revision: str | None = None,
) -> Path:
    """Ensure one Distribution Release is installed at ``<workspace>/official-release/``.

    Calls ``init_workspace`` first, so *workspace* need not already exist.
    Repository and token are never caller-supplied: the repository is this
    library's ``DISTRIBUTION_RELEASE_REPOSITORY``, which is public, and the
    optional token is read from the ``HF_TOKEN`` environment variable
    (anonymous, ``token=False``, when unset).

    With ``revision=None``, this ensures *some* release is installed: if
    ``official-release/`` already holds one (fetched by this call, pre-installed
    by a test harness, or relayouted by hand), that tree is left exactly as it
    is and this is a no-op; otherwise the latest published release (the
    repository's default branch) is installed and its manifest records the
    version it resolved to. With an explicit ``revision``, that revision is ensured:
    installed if absent, left alone if it is already installed, and otherwise
    ``official-release/`` is replaced whole (removed, then reinstalled) — a
    local run against the old revision is never mixed with the new one's
    shipped Results. There is no workspace-level fetch manifest: the
    installed tree identifies itself. "Already installed" means *revision*
    equals the ``version`` in ``official-release/release.json`` (a fetch by
    tag) or the commit this function recorded in
    ``official-release/.cache/commit`` when it installed the tree (a fetch by
    commit). A tree with no such record is not known to be at any commit, so
    a fetch by commit reinstalls it once.

    Args:
        workspace: The workspace to fetch into.
        revision: The tag or commit to install, or ``None`` to ensure any
            release is present, installing the latest only if none is.

    Returns:
        *workspace*, as a ``Path``.

    Raises:
        ValueError: If ``workspace`` exists and is not a workspace, or for the
            reasons ``init_workspace``/the underlying fetch raise.

    Example:
        >>> workspace = fetch_benchmark("my-workspace")  # doctest: +SKIP
    """
    root = init_workspace(workspace)
    release_root = workspace_release_root(root)
    if revision is None:
        if (release_root / RELEASE_MANIFEST_NAME).is_file():
            return root
        target_revision = LATEST_RELEASE_REVISION
    else:
        target_revision = revision
        identity = read_release_identity(release_root)
        record = release_root / INSTALLED_COMMIT_RECORD
        if identity is not None and target_revision in {
            identity.version,
            record.read_text() if record.is_file() else None,
        }:
            return root
    token: str | Literal[False] = os.environ.get("HF_TOKEN") or False
    staging = release_root.parent / f".{release_root.name}-staging"
    if staging.exists() or staging.is_symlink():
        shutil.rmtree(staging, ignore_errors=True)
    commit = _fetch_release_tree(
        DISTRIBUTION_RELEASE_REPOSITORY,
        revision=target_revision,
        destination=staging,
        token=token,
    )
    record = staging / INSTALLED_COMMIT_RECORD
    record.parent.mkdir()
    record.write_text(commit)
    if release_root.is_symlink():
        release_root.unlink()
    elif release_root.exists():
        shutil.rmtree(release_root)
    staging.rename(release_root)
    return root
