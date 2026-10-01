"""scripts/fetch_corpus.py against an in-memory GitHub tarball (no network)."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_corpus.py"
_spec = importlib.util.spec_from_file_location("fetch_corpus", SCRIPT)
fetch_corpus = importlib.util.module_from_spec(_spec)
sys.modules["fetch_corpus"] = fetch_corpus  # dataclasses look their module up here
_spec.loader.exec_module(fetch_corpus)

UPSTREAM = {
    "LICENSE": "MIT License\n\nCopyright (c) Example\n",
    "README.md": "# Example\n",  # outside the subdir: never fetched
    "docs/index.md": "# Example\n\nTop page.\n",
    "docs/guide/install.md": "# Install\n\n{* ../../docs_src/app.py hl[3] *}\n\nRun it.\n",
    "docs/guide/logo.png": "not markdown",
    "docs/release-notes.md": "# Release notes\n",
    "docs/about/team.md": "# Team\n",
}
SOURCE = {
    "dest": "example",
    "repo": "example/example",
    "tag": "v1.0.0",
    "commit": "0123abcd" * 5,
    "license": "MIT",
    "subdir": "docs",
    "include": ["**/*.md"],
    "exclude": ["release-notes.md", "about/**"],
    "drop_lines": ["^\\{\\* .* \\*\\}$"],
}


def _tarball(files: dict[str, str], top: str = "example-0123abcd") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode("utf-8")
            info = tarfile.TarInfo(f"{top}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture
def env(tmp_path):
    corpora, raw = tmp_path / "corpora", tmp_path / "raw"
    corpora.mkdir()
    calls: list[str] = []
    upstream = {"archive": _tarball(UPSTREAM)}

    def pin(files_sha256: str = "") -> None:
        manifest = {"name": "demo", "sources": [SOURCE], "files_sha256": files_sha256}
        (corpora / "demo.json").write_text(json.dumps(manifest), encoding="utf-8")

    def fetch(url: str) -> bytes:
        calls.append(url)
        return upstream["archive"]

    def run(*args: str) -> int:
        return fetch_corpus.main(["demo", *args], corpora_dir=corpora, raw_dir=raw, fetch=fetch)

    pin()
    return SimpleNamespace(
        run=run,
        pin=pin,
        calls=calls,
        upstream=upstream,
        corpus=raw / "demo",
        notice=raw / "demo.NOTICE.md",
        manifest=lambda: json.loads((corpora / "demo.json").read_text(encoding="utf-8")),
    )


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("**/*.md", "index.md", True),
        ("**/*.md", "guide/install.md", True),
        ("**/*.md", "guide/logo.png", False),
        ("*.md", "guide/install.md", False),
        ("release-notes.md", "release-notes.md", True),
        ("release-notes.md", "guide/release-notes.md", False),
        ("about/**", "about/team.md", True),
        ("about/**", "aboutus.md", False),
    ],
)
def test_globs_span_folders_only_with_a_double_star(pattern, path, expected) -> None:
    assert bool(fetch_corpus.glob_regex(pattern).match(path)) is expected


def test_update_hash_installs_the_selected_files_and_pins_them(env) -> None:
    assert env.run("--update-hash") == 0

    installed = fetch_corpus.installed_files(env.corpus)
    # The subdir is stripped; the png, release notes and about/ are left out.
    assert sorted(installed) == ["example/guide/install.md", "example/index.md"]
    assert env.calls == [f"https://codeload.github.com/example/example/tar.gz/{SOURCE['commit']}"]
    assert env.manifest()["files_sha256"] == fetch_corpus.files_sha256(installed)


def test_drop_lines_removes_only_the_matching_lines(env) -> None:
    env.run("--update-hash")

    page = (env.corpus / "example" / "guide" / "install.md").read_text(encoding="utf-8")
    assert page == "# Install\n\n\nRun it.\n"


def test_the_notice_sits_beside_the_corpus_and_carries_the_license(env) -> None:
    env.run("--update-hash")

    notice = env.notice.read_text(encoding="utf-8")
    assert "[example/example](https://github.com/example/example) | v1.0.0" in notice
    assert "1 line matching" in notice
    assert "MIT License\n\nCopyright (c) Example" in notice
    assert not any(path.name == "demo.NOTICE.md" for path in env.corpus.rglob("*"))


def test_a_second_run_is_a_no_op(env, capsys) -> None:
    env.run("--update-hash")
    env.calls.clear()

    assert env.run() == 0

    assert env.calls == []
    assert "up to date" in capsys.readouterr().out


def test_a_hash_mismatch_installs_nothing(env, capsys) -> None:
    env.pin("0" * 64)

    assert env.run() == 1

    assert not env.corpus.exists()
    assert "--update-hash" in capsys.readouterr().err


def test_a_damaged_corpus_is_fetched_again(env) -> None:
    env.run("--update-hash")
    page = env.corpus / "example" / "index.md"
    page.write_text("edited by hand", encoding="utf-8")

    assert env.run() == 0  # no longer matches the pin, so it's fetched and verified again

    assert page.read_text(encoding="utf-8") == "# Example\n\nTop page.\n"


def test_upstream_drift_leaves_the_installed_corpus_alone(env) -> None:
    env.run("--update-hash")
    page = env.corpus / "example" / "index.md"
    page.write_text("edited by hand", encoding="utf-8")
    env.upstream["archive"] = _tarball({**UPSTREAM, "docs/index.md": "# Example\n\nChanged.\n"})

    assert env.run() == 1  # what upstream serves now hashes differently from the pin

    assert page.read_text(encoding="utf-8") == "edited by hand"


def test_paths_that_climb_out_of_the_archive_are_skipped() -> None:
    archive = _tarball({"docs/../../escape.md": "# Out\n", "docs/ok.md": "# In\n"})

    extracted = fetch_corpus.extract(archive, {**SOURCE, "exclude": []})

    assert sorted(extracted.files) == ["example/ok.md"]
