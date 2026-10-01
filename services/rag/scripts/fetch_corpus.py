#!/usr/bin/env python3
"""Fetch a pinned public docs corpus into data/raw/<name>/.

    python3 scripts/fetch_corpus.py toolchain_docs                # fetch, or "up to date"
    python3 scripts/fetch_corpus.py toolchain_docs --update-hash  # after editing the sources

The manifest, corpora/<name>.json (committed), pins every source to a release
tag and its commit. For each source this downloads GitHub's tarball of that
commit, keeps the files under ``subdir`` that match ``include`` and none of
``exclude`` (globs relative to ``subdir``; ``**`` spans folders), removes the
lines matching any ``drop_lines`` pattern, and writes the result to
data/raw/<name>/<dest>/. The corpus is installed only if the sha256 of its
files equals the manifest's ``files_sha256``; ``--update-hash`` records the new
value instead. data/raw/<name>.NOTICE.md lists the sources, what was changed
and the licenses, in full. It sits beside the corpus, not inside it, so an
ingest of the corpus never indexes it.

Standard library only, and nothing from ``rag``: it runs on the host with any
Python 3.10 or newer, so the files belong to you rather than to a container's root.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import re
import shutil
import sys
import tarfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

SERVICE_ROOT = Path(__file__).resolve().parents[1]
CORPORA_DIR = SERVICE_ROOT / "corpora"
RAW_DIR = SERVICE_ROOT / "data" / "raw"
TARBALL_URL = "https://codeload.github.com/{repo}/tar.gz/{commit}"
# Top-level files that carry a project's license (LICENSE, LICENSE-MIT, COPYING...).
LICENSE_FILE = re.compile(r"(LICEN[CS]E|COPYING)([-._][\w.-]*)?", re.IGNORECASE)


@dataclass
class Extracted:
    """One source's share of the corpus."""

    files: dict[str, bytes] = field(default_factory=dict)  # "<dest>/<path>" -> content
    licenses: dict[str, str] = field(default_factory=dict)  # file name -> text
    dropped_lines: int = 0


@functools.cache
def glob_regex(pattern: str) -> re.Pattern[str]:
    """``**/`` matches zero or more folders; ``*`` and ``?`` stay within one name."""
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def selected(path: str, include: list[str], exclude: list[str]) -> bool:
    return any(glob_regex(p).match(path) for p in include) and not any(
        glob_regex(p).match(path) for p in exclude
    )


def drop_matching_lines(data: bytes, patterns: list[re.Pattern[str]]) -> tuple[bytes, int]:
    lines = data.decode("utf-8").splitlines(keepends=True)
    kept = [line for line in lines if not any(p.search(line.rstrip("\r\n")) for p in patterns)]
    return "".join(kept).encode("utf-8"), len(lines) - len(kept)


def extract(archive: bytes, source: dict) -> Extracted:
    """The selected files of one source, from its tarball (top folder stripped)."""
    subdir = source["subdir"].strip("/")
    include, exclude = source["include"], source.get("exclude", [])
    drops = [re.compile(p) for p in source.get("drop_lines", [])]
    out = Extracted()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar:
            parts = PurePosixPath(member.name).parts[1:]  # drop "<repo>-<commit>/"
            if not member.isfile() or not parts or ".." in parts:
                continue
            if len(parts) == 1 and LICENSE_FILE.fullmatch(parts[0]):
                out.licenses[parts[0]] = tar.extractfile(member).read().decode("utf-8")
                continue
            path = "/".join(parts)
            if subdir:
                if not path.startswith(subdir + "/"):
                    continue
                path = path[len(subdir) + 1 :]
            if not selected(path, include, exclude):
                continue
            data = tar.extractfile(member).read()
            if drops:
                data, dropped = drop_matching_lines(data, drops)
                out.dropped_lines += dropped
            out.files[f"{source['dest']}/{path}"] = data
    return out


def files_sha256(files: dict[str, bytes]) -> str:
    """sha256 of the sorted "path<TAB>sha256" lines, one per file."""
    lines = "".join(
        f"{path}\t{hashlib.sha256(data).hexdigest()}\n" for path, data in sorted(files.items())
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def installed_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "groundwork-fetch-corpus"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def notice(manifest: dict, parts: list[tuple[dict, Extracted]]) -> str:
    name = manifest["name"]
    lines = [
        f"# {name}: third-party documentation",
        "",
        f"The files in `{name}/` are copies of other projects' documentation, fetched by",
        f"`scripts/fetch_corpus.py` from `corpora/{name}.json` and used only as a retrieval",
        "corpus. Each project's license applies to its files; the texts follow.",
        "",
        "| Folder | Project | Release | Commit | License | Files |",
        "|---|---|---|---|---|---|",
    ]
    for source, part in parts:
        lines.append(
            f"| `{source['dest']}/` | [{source['repo']}](https://github.com/{source['repo']}) "
            f"| {source['tag']} | `{source['commit'][:12]}` | {source['license']} "
            f"| {len(part.files)} |"
        )
    lines += ["", "## Changes", ""]
    for source, part in parts:
        change = (
            f"- `{source['dest']}/`: the files under `{source['subdir'].strip('/')}/` matching "
            + ", ".join(f"`{p}`" for p in source["include"])
        )
        if source.get("exclude"):
            change += ", except " + ", ".join(f"`{p}`" for p in source["exclude"])
        if source.get("drop_lines"):
            patterns = ", ".join(f"`{p}`" for p in source["drop_lines"])
            lines_word = "line" if part.dropped_lines == 1 else "lines"
            change += f"; {part.dropped_lines} {lines_word} matching {patterns} removed"
        lines.append(change + ".")
    lines += ["", "Nothing else was changed.", "", "## Licenses"]
    for source, part in parts:
        for file_name, text in sorted(part.licenses.items()):
            lines += ["", f"### {source['repo']}: {file_name}", "", "```text", text.rstrip(), "```"]
    return "\n".join(lines) + "\n"


def install(files: dict[str, bytes], target: Path) -> None:
    """Write the corpus beside ``target``, then swap it into place."""
    staging = target.with_name(f".{target.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    for path, data in files.items():
        out = staging / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    if target.exists():
        shutil.rmtree(target)
    staging.rename(target)


def main(
    argv: list[str] | None = None,
    *,
    corpora_dir: Path = CORPORA_DIR,
    raw_dir: Path = RAW_DIR,
    fetch: Callable[[str], bytes] = download,
) -> int:
    parser = argparse.ArgumentParser(description="Fetch a pinned public docs corpus.")
    parser.add_argument("name", help="corpus manifest name, e.g. toolchain_docs")
    parser.add_argument(
        "--update-hash",
        action="store_true",
        help="fetch even if up to date, and write the result's hash into the manifest",
    )
    args = parser.parse_args(argv)

    manifest_path = corpora_dir / f"{args.name}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    name, pinned = manifest["name"], manifest.get("files_sha256", "")
    target = raw_dir / name

    if not args.update_hash and target.is_dir():
        present = installed_files(target)
        if files_sha256(present) == pinned:
            print(
                f"{name}: up to date ({len(present)} files, sha256 {pinned[:12]}); "
                "after editing the sources, run with --update-hash"
            )
            return 0

    parts: list[tuple[dict, Extracted]] = []
    files: dict[str, bytes] = {}
    for source in manifest["sources"]:
        url = TARBALL_URL.format(repo=source["repo"], commit=source["commit"])
        print(f"fetching {source['repo']} {source['tag']} ({source['commit'][:12]})")
        try:
            part = extract(fetch(url), source)
        except OSError as exc:
            print(f"error: downloading {url} failed: {exc}", file=sys.stderr)
            return 1
        if not part.files:
            print(f"error: no files of {source['repo']} matched the manifest", file=sys.stderr)
            return 1
        parts.append((source, part))
        files.update(part.files)

    digest = files_sha256(files)
    if args.update_hash:
        manifest["files_sha256"] = digest
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"{manifest_path.name}: files_sha256 set to {digest}")
    elif digest != pinned:
        print(
            f"error: {name}: the fetched files hash to {digest}, but the manifest pins "
            f"{pinned or '(nothing)'}. Nothing was installed; if the sources changed on "
            "purpose, run again with --update-hash.",
            file=sys.stderr,
        )
        return 1

    raw_dir.mkdir(parents=True, exist_ok=True)
    install(files, target)
    (raw_dir / f"{name}.NOTICE.md").write_text(notice(manifest, parts), encoding="utf-8")
    print(f"{name}: installed {len(files)} files in {target} (sha256 {digest[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
