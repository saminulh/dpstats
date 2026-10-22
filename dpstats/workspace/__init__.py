"""The on-disk world: a workspace, and a Task's view of it.

A workspace is any directory holding ``local/`` (the user's own store; every
write lands here) and optionally ``official-release/`` (a fetched Distribution
Release, byte for byte, never written after fetch). ``layout`` is the sole
home of that path vocabulary; ``fetch`` installs a release into one; a Task
collection, opened with :func:`open_collection`, is one Task's view across
both trees.
"""

from dpstats.workspace.collection import (
    TaskCollection as TaskCollection,
    open_collection as open_collection,
)
from dpstats.workspace.fetch import fetch_benchmark as fetch_benchmark
from dpstats.workspace.layout import (
    ReleaseIdentity as ReleaseIdentity,
    init_workspace as init_workspace,
)
