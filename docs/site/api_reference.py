"""Assemble shared API directories and their native object pages."""

from __future__ import annotations

import json
import posixpath
import re
from importlib import import_module
from pathlib import Path
from textwrap import indent
from typing import Annotated, Any, get_origin

import griffe
import yaml

from dpstats.algorithm import BaseAlgorithm, Transformation
from dpstats.parameters import RangeSearch, ValuesSearch


def object_route(identifier: str, routes: dict[str, str]) -> str:
    """Find the generated page owning an object or its member."""
    return routes[
        max(
            (
                name
                for name in routes
                if identifier == name or identifier.startswith(name + ".")
            ),
            key=len,
        )
    ]


def object_fragment(identifier: str, routes: dict[str, str]) -> str:
    """The anchor an object has on its page, where the page holds several.

    A generated API page is named by the object path it renders, so the
    object's own anchor is that name. A class's own page is named by the class
    and renders nothing else, so it has no fragment.
    """
    owner = max(
        (
            name
            for name in routes
            if identifier == name or identifier.startswith(name + ".")
        ),
        key=len,
    )
    basename = posixpath.basename(routes[owner]).removesuffix(".md")
    if basename != owner:
        return ""
    return basename + identifier[len(owner) :]


def rewrite_links(
    text: str,
    page: str,
    groups: dict[str, Path],
    routes: dict[str, str],
) -> str:
    """Repoint Markdown links after shared reference pages become directories."""

    def replace(match: re.Match[str]) -> str:
        link = match[1]
        if "://" in link or link.startswith("/"):
            return match[0]
        path, marker, fragment = link.partition("#")
        resolved = (
            posixpath.normpath(posixpath.join(posixpath.dirname(page), path))
            if path
            else page
        )
        group = resolved.removeprefix("library/reference/").removesuffix(".md")
        if group in groups and resolved == f"library/reference/{group}.md":
            target = f"library/reference/{group}/index.md"
        else:
            return match[0]
        if fragment and any(
            fragment == name or fragment.startswith(name + ".") for name in routes
        ):
            target = object_route(fragment, routes)
            fragment = object_fragment(fragment, routes)
        target = posixpath.relpath(target, posixpath.dirname(page))
        return "](" + target + (marker + fragment if marker else "") + ")"

    return re.sub(r"\]\(([^\s)]+)\)", replace, text)


def reference_pages(
    groups: dict[str, Path],
) -> tuple[dict[str, str], dict[str, str]]:
    """Derive pages from the standalone reference's module directives.

    Selection matches the module's own curated surface: a member is a page
    candidate only while ``obj.is_public`` (Griffe's ``__all__``-membership
    check where a module declares one, its non-underscore/non-imported
    fallback otherwise) and it carries a docstring. A function or class
    that is exported through ``__all__`` but undocumented fails the build
    instead of silently vanishing from the reference. Public annotated type
    aliases also receive pages so rendered annotations have API destinations;
    other undocumented constants are skipped.
    """
    loader = griffe.GriffeLoader()
    loader.load("dpstats")
    get_module = loader.modules_collection.get_member

    def is_page_candidate(module: Any, obj: Any) -> bool:
        """Keep a public, documented member; griffe types both args ``Any``.

        Raises when ``obj`` is named in ``module``'s own ``__all__`` but
        undocumented, rather than silently dropping a declared export.
        """
        if not obj.is_public or obj.is_module:
            return False
        if obj.has_docstrings:
            return True
        if (
            obj.is_attribute
            and get_origin(getattr(import_module(module.path), obj.name, None))
            is Annotated
        ):
            return True
        exported = module.exports is not None and obj.name in module.exports
        if exported and (obj.is_function or obj.is_class):
            raise ValueError(
                f"{module.path}.{obj.name} is exported via __all__ but "
                "has no docstring."
            )
        return False

    directives = {
        group: list(
            re.finditer(
                r"^::: ([\w.]+)\n((?:[ \t]+[^\n]*\n)*)", source.read_text(), re.M
            )
        )
        for group, source in groups.items()
    }
    modules = {
        group: [get_module(match[1]) for match in matches]
        for group, matches in directives.items()
    }
    options = {
        (group, match[1]): (yaml.safe_load(match[2]) or {}).get("options", {})
        for group, matches in directives.items()
        for match in matches
    }
    members = {}
    owners = {}
    for group, declarations in modules.items():
        for module in declarations:
            selected = [
                obj for obj in module.members.values() if is_page_candidate(module, obj)
            ]
            names = options[group, module.path].get("members")
            if names is not None:
                missing = set(names) - {obj.name for obj in selected}
                if missing:
                    raise ValueError(
                        f"{group}: undocumented or nonpublic members: {sorted(missing)}"
                    )
                selected = [obj for obj in selected if obj.name in names]
            members[group, module.path] = selected
            for obj in selected:
                previous = owners.get(obj.canonical_path)
                if previous is None or (
                    previous[1].is_alias
                    and (not obj.is_alias or len(obj.path) > len(previous[1].path))
                ):
                    owners[obj.canonical_path] = (group, obj)
    routes = {
        obj.path: f"library/reference/{owners[obj.canonical_path][0]}/{owners[obj.canonical_path][1].path}.md"
        for selected in members.values()
        for obj in selected
    }
    pages = {}
    for group, declarations in modules.items():
        prefix = f"library/reference/{group}"
        directory = groups[group].read_text()
        title = re.search(r"^# (.+)$", directory, re.M)[1]
        nav = [f"- [{title}](index.md)"]

        # Existing relative prose links gain one directory level; fragments keep
        # their authored section meaning unless they identify an API object.
        def move_link(match: re.Match[str]) -> str:
            href = match[1]
            if "://" in href or href.startswith("/"):
                return match[0]
            path, sep, fragment = href.partition("#")
            if fragment and any(
                fragment == name or fragment.startswith(name + ".") for name in routes
            ):
                target = object_route(fragment, routes)
                fragment = object_fragment(fragment, routes)
            elif path:
                target = posixpath.normpath(posixpath.join("library/reference", path))
                other = target.removeprefix("library/reference/").removesuffix(".md")
                if other in groups:
                    target = f"library/reference/{other}/index.md"
            else:
                return match[0]
            return (
                "]("
                + posixpath.relpath(target, prefix)
                + (sep + fragment if sep else "")
                + ")"
            )

        directory = re.sub(r"\]\(([^\s)]+)\)", move_link, directory)
        for module, directive in zip(declarations, directives[group]):
            selected = members[group, module.path]
            authored_options = options[group, module.path]
            label = authored_options.get("heading")
            entries = [
                f"    - [{obj.name}]({posixpath.relpath(routes[obj.path], prefix)})"
                for obj in selected
                if owners[obj.canonical_path][0] == group
                and owners[obj.canonical_path][1].path == obj.path
            ]
            if entries:
                if len(declarations) == 1:
                    nav.extend(entry.removeprefix("    ") for entry in entries)
                else:
                    nav.extend([f"- {label or module.path}", *entries])
            rendered_options = {
                **authored_options,
                "heading_level": 2,
                "summary": True,
                "members": [obj.name for obj in selected],
                "extra": {"directory_only": True},
            }
            if label:
                rendered_options["toc_label"] = label
            if len(declarations) == 1:
                rendered_options["show_root_heading"] = False
            replacement = f"::: {module.path}\n" + indent(
                yaml.safe_dump({"options": rendered_options}, sort_keys=False), "    "
            )
            directory = directory.replace(directive[0], replacement)
        pages[f"{prefix}/index.md"] = (
            f"---\ntitle: {json.dumps(title)}\n---\n\n" + directory
        )
        pages[f"{prefix}/SUMMARY.md"] = "\n".join(nav) + "\n"
    for group, obj in owners.values():
        prefix = f"library/reference/{group}"
        options = [
            "      heading_level: 1",
            "      summary: true",
            "      show_if_no_docstring: true",
        ]
        module, _, name = obj.path.rpartition(".")
        pages[f"{prefix}/{obj.path}.md"] = (
            f"---\ntitle: {obj.name}\n---\n\n"
            + f"```python\nfrom {module} import {name}\n```\n\n"
            + f"::: {obj.path}\n    options:\n"
            + "\n".join(options)
            + "\n"
        )
    # Prepared configuration provenance carries canonical class paths, including
    # classes documented through a public re-export.
    routes.update({obj.canonical_path: routes[obj.path] for _, obj in owners.values()})
    return pages, routes


def configured_base(model: type) -> type | None:
    """Return the platform base whose own fields are not this class's configuration.

    Shared with the algorithm-provenance popover generator, which reads the
    same configured fields for a displayed result's exact component classes.
    """
    if issubclass(model, BaseAlgorithm):
        return BaseAlgorithm
    if issubclass(model, Transformation):
        return Transformation
    return None


def format_search(search: RangeSearch | ValuesSearch | None) -> str:
    """Render a declared tuning-search policy as short human-readable text."""
    if search is None:
        return ""
    if isinstance(search, ValuesSearch):
        return ", ".join(repr(value) for value in search.values)
    return f"{search.lower} – {search.upper} ({search.scale} scale)"
