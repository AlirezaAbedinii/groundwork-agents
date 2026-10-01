"""Source names are paths relative to the ingested folder (deterministic, no network)."""
from __future__ import annotations

import json

from rag.config import Settings
from rag.ingestion import build_chunks_for_dir, build_chunks_for_file

# The Ferry corpus's chunk ids as computed at 8ce0e61, before source names became
# relative paths. Ferry is one flat folder, so each relative path is the file name
# and none of these may change.
FERRY_CHUNK_IDS = [
    "04927107a329a358", "050b19d6a2360ad8", "06838cfc1654895d", "07e759f975f3de5d",
    "0a14b39547ad3781", "1108860ad30789ba", "1908c866850e6101", "1c9a6dc6a78f64f2",
    "1e75de0558e698d9", "227c619d97bc16c8", "278d593a9b0b8f3c", "29f51bf95a40a971",
    "2c7620b5698b5248", "3b3efa5f2d1fc20c", "4791b736251ecd3c", "4b7b0e1df3661bd7",
    "54dc6126ca72a3af", "5e61abe3eb5f8706", "6209be453339fe6b", "6bf097cc136f5db3",
    "6cf26fd88f386660", "80f8b0107f0fd0ad", "88b936bac0c61333", "8c068304202040bb",
    "9889b7b0d46b9c9f", "a5912021050126f1", "ac17ee8053d20d78", "bb3c9a677e51e551",
    "c69576d2bc5aa685", "d04a764ab84b3e63", "d7aa9a3e40291e04", "e14ccea5fa80e30f",
    "e15f2afe7a5597a9", "f6fa124aa08a391b", "fa7d0bf0ddba4840", "fc62937662d0aae3",
    "fe5a583b2b4ee26f",
]

PAGE = "# Install\n\nRun the installer, then check the version.\n"


def _settings(tmp_path) -> Settings:
    return Settings(_env_file=None, data_processed_dir=tmp_path / "processed")


def _corpus(tmp_path):
    root = tmp_path / "corpus"
    for name in ("uv/index.md", "pydantic/index.md", "top.md"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(PAGE, encoding="utf-8")
    return root


def test_nested_folders_give_relative_source_names(tmp_path) -> None:
    chunks = build_chunks_for_dir(_corpus(tmp_path), settings=_settings(tmp_path))

    by_source = {c.source_file: c for c in chunks}
    assert set(by_source) == {"pydantic/index.md", "top.md", "uv/index.md"}
    # Same text, different pages: the ids hash the relative path, so they differ.
    assert by_source["uv/index.md"].chunk_id != by_source["pydantic/index.md"].chunk_id


def test_a_single_file_keeps_its_name(tmp_path) -> None:
    page = _corpus(tmp_path) / "uv" / "index.md"

    chunks = build_chunks_for_file(page, settings=_settings(tmp_path))

    assert {c.source_file for c in chunks} == {"index.md"}


def test_processed_files_of_same_named_pages_dont_collide(tmp_path) -> None:
    build_chunks_for_dir(_corpus(tmp_path), settings=_settings(tmp_path), persist=True)

    written = sorted(p.name for p in (tmp_path / "processed").iterdir())
    assert written == ["pydantic__index.json", "top.json", "uv__index.json"]
    saved = json.loads((tmp_path / "processed" / "uv__index.json").read_text(encoding="utf-8"))
    assert saved["source_file"] == "uv/index.md"


def test_the_ferry_chunk_ids_are_unchanged() -> None:
    settings = Settings(_env_file=None)

    chunks = build_chunks_for_dir(settings.corpus_dir, settings=settings)

    assert sorted(c.chunk_id for c in chunks) == FERRY_CHUNK_IDS
