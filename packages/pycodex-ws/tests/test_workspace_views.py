import base64
import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import pytest
from PIL import Image
from starlette.testclient import TestClient

import workspace_server.app as workspace_app
from pycodex import AssistantMessage, ModelResponse
from pycodex.bootstrap import get_tools
from tests.fakes import ScriptedModelClient
from test_workspace_server import make_session, wait_snapshot
from workspace_server import (
    WorkspaceDefinition,
    WorkspaceEntry,
    WorkspaceRegistry,
    WorkspaceSessionManager,
    build_parser,
    create_app,
    create_multi_workspace_app,
)
from workspace_server.app import _build_workspace_entry
from workspace_server.workspace_views import WorkspaceViewOpener, WorkspaceViewStore


def view_link(action, **parameters):
    def path(value):
        return quote(str(value)).replace("..", "%2E%2E")

    def target(value):
        revision, separator, name = value.partition(":")
        return quote(revision) + separator + path(name)

    if action == "file":
        href = (
            "git:" + quote(parameters["rev"]) + ":" + path(parameters["path"])
            if "rev" in parameters else path(parameters["path"])
        )
    elif action == "diff":
        href = "diff:" + path(parameters["left"]) + ".." + path(parameters["right"])
    elif "left" in parameters:
        href = "diff:" + target(parameters["left"]) + ".." + target(parameters["right"])
    else:
        revision = quote(parameters["base"])
        if "head" in parameters:
            revision += ".." + quote(parameters["head"])
        href = "diff:" + revision + ":" + path(parameters["path"])
    if "line" in parameters:
        side = "R" if action != "file" and parameters.get("side") != "left" else "L"
        href += "#" + side + str(parameters["line"])
    return href


def git(directory, *arguments):
    return (
        subprocess.check_output(
            [
                "git", "-C", str(directory),
                "-c", "user.name=Test",
                "-c", "user.email=test@example.invalid",
                "-c", "commit.gpgsign=false",
                "-c", "core.hooksPath=" + os.devnull,
            ] + list(arguments),
            stderr=subprocess.PIPE,
        )
        .decode("utf-8")
        .strip()
    )


@pytest.fixture
def git_repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "data").mkdir()
    (repo / "data" / "A.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "old name.py").write_text("old\n", encoding="utf-8")
    (repo / "deleted.py").write_text("removed\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "first")
    first = git(repo, "rev-parse", "HEAD")
    (repo / "data" / "A.py").write_text("value = 2\n", encoding="utf-8")
    (repo / "old name.py").rename(repo / "new name.py")
    (repo / "deleted.py").unlink()
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "second")
    second = git(repo, "rev-parse", "HEAD")
    return repo, first, second


def test_file_links_reuse_tabs_and_restore_sources(tmp_path):
    work_dir = tmp_path / "project"
    work_dir.mkdir()
    source = tmp_path / "A file.py"
    source.write_text("old ≤ value\n", encoding="utf-8")
    board = tmp_path / "board.html"
    board.write_text("live board", encoding="utf-8")
    views = WorkspaceViewStore(board.with_suffix(".pycodex-views.json"))
    opener = WorkspaceViewOpener(views, work_dir)
    opened = opener.open("../A%20file.py")
    source.write_text("new value\n", encoding="utf-8")
    selected = opener.open(view_link("file", path=str(source)))
    assert len(views.list_views()) == 1
    assert selected["id"] == opened["id"]
    assert selected["title"] == "../A file.py"
    assert selected["reveal_revision"] > opened["reveal_revision"]
    assert selected["documents"][0]["text"] == "new value\n"
    assert "documents" not in views.list_views()[0]
    assert views.list_views()[0]["live"]
    source.write_text("updated while closed\n", encoding="utf-8")
    restored = WorkspaceViewStore(board.with_suffix(".pycodex-views.json"))
    assert restored.get(opened["id"])["documents"][0]["text"] == "updated while closed\n"
    source.unlink()
    missing = WorkspaceViewOpener(restored, work_dir).open(quote(str(source)))
    assert missing["id"] == opened["id"]
    document = missing["documents"][0]
    assert document["text"] == "updated while closed\n"
    assert document["missing"] and "no longer exists" in document["error"]
    source.write_text("recreated\n", encoding="utf-8")
    assert restored.get(opened["id"])["documents"][0]["text"] == "recreated\n"
    assert "error" not in restored.get(opened["id"])["documents"][0]
    restored.close(opened["id"])
    assert WorkspaceViewStore(board.with_suffix(".pycodex-views.json")).list_views() == []
    assert board.read_text(encoding="utf-8") == "live board"
    if os.name != "nt":
        assert restored.path.stat().st_mode & 0o077 == 0


def test_tabs_use_complete_paths_and_ordered_comparisons(tmp_path):
    for directory in ("data", "other"):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "A.py").write_text(directory + "\n", encoding="utf-8")
    views = WorkspaceViewStore()
    opener = WorkspaceViewOpener(views, tmp_path)
    for href in (
        "data/A.py", "other/A.py",
        view_link("diff", left="data/A.py", right="other/A.py"),
        view_link("diff", left="other/A.py", right="data/A.py"),
    ):
        opener.open(href)
    opened = views.list_views()
    assert [view["title"] for view in opened] == [
        "data/A.py", "other/A.py",
        "data/A.py <-> other/A.py", "other/A.py <-> data/A.py",
    ]
    assert opener.open("./data/../data/A.py")["id"] == opened[0]["id"]
    selected = opener.open(view_link(
        "diff", left=str(tmp_path / "data" / "A.py"), right="./other/A.py"
    ))
    assert selected["id"] == opened[2]["id"]
    assert [view["id"] for view in views.list_views()] == [view["id"] for view in opened]
    assert views.list_views()[2]["reveal_revision"] > views.list_views()[3]["reveal_revision"]


@pytest.mark.parametrize("name", ["空 格 # % & +.py", "README", "123", "literal%20.py"])
def test_encoded_file_paths_are_decoded_once(tmp_path, name):
    (tmp_path / name).write_text("first\nsecond", encoding="utf-8")
    views = WorkspaceViewStore()
    opener = WorkspaceViewOpener(views, tmp_path)
    opened = opener.open(quote(name) + "#L2")
    selected = opener.open(view_link("file", path=name, line=1))
    assert selected["id"] == opened["id"]
    assert opened["title"] == name
    assert opened["selection"] == {"document": 0, "line": 2}
    assert selected["selection"] == {"document": 0, "line": 1}


def test_uri_path_keeps_literal_numeric_suffix(tmp_path):
    if os.name == "nt":
        pytest.skip("Windows filenames cannot contain colons")
    (tmp_path / "A.py:42").write_text("literal colon\n", encoding="utf-8")
    view = WorkspaceViewOpener(WorkspaceViewStore(), tmp_path).open(
        view_link("file", path="A.py:42", line=1)
    )
    assert view["title"] == "A.py:42"
    assert view["selection"] == {"document": 0, "line": 1}


def test_native_diff_paths_keep_parent_components_and_encoded_separators(tmp_path):
    work_dir = tmp_path / "nested"
    work_dir.mkdir()
    (tmp_path / "old.py").write_text("old\n", encoding="utf-8")
    (tmp_path / "new.py").write_text("new\n", encoding="utf-8")
    (work_dir / "old..file.py").write_text("literal dots\n", encoding="utf-8")
    views = WorkspaceViewStore()
    opener = WorkspaceViewOpener(views, work_dir)
    view = opener.open("diff:../old.py..../new.py#R1")
    assert [document["text"] for document in view["documents"]] == ["old\n", "new\n"]
    assert view["selection"] == {"document": 1, "line": 1}
    literal = opener.open("diff:old%2E%2Efile.py..../new.py#L1")
    assert literal["documents"][0]["text"] == "literal dots\n"
    assert literal["title"] == "old..file.py <-> ../new.py"


def test_windows_drive_links_keep_the_drive_and_line(tmp_path):
    opener = WorkspaceViewOpener(WorkspaceViewStore(), tmp_path)
    for href, path in [
        ("C:/project/A.py:42", "C:/project/A.py"),
        ("C:%5Cproject%5CA.py#L42", "C:\\project\\A.py"),
    ]:
        action, options = opener._parse_link(href)
        assert action == "file"
        assert options == {"path": path, "line": "42"}


def test_file_line_selection_reuses_tabs_and_reads_current_contents(tmp_path):
    source = tmp_path / "A file.py"
    source.write_bytes(b"first\r\nsecond\rthird\n")
    board = tmp_path / "board.html"
    views = WorkspaceViewStore(board.with_suffix(".pycodex-views.json"))
    opener = WorkspaceViewOpener(views, tmp_path)
    opened = opener.open("A%20file.py:2")
    assert opened["selection"] == {"document": 0, "line": 2}
    selected = opener.open(quote(str(source)) + "#L3")
    assert selected["id"] == opened["id"]
    assert selected["revision"] == opened["revision"]
    assert selected["reveal_revision"] > opened["reveal_revision"]
    assert selected["selection"] == {"document": 0, "line": 3}
    restored = WorkspaceViewStore(board.with_suffix(".pycodex-views.json"))
    assert restored.list_views()[0]["selection"] == selected["selection"]
    source.write_text("first\nsecond\nthird\nfourth\nfifth", encoding="utf-8")
    opener = WorkspaceViewOpener(restored, tmp_path)
    assert opener.open("A%20file.py:5")["selection"] == {"document": 0, "line": 5}
    assert opener.open("A%20file.py")["selection"] is None
    assert [view["id"] for view in restored.list_views()] == [opened["id"]]


def test_diff_line_selection_targets_either_side_without_changing_identity(tmp_path):
    for name in ("A.py", "B.py"):
        (tmp_path / name).write_text("first\nsecond\nthird", encoding="utf-8")
    views = WorkspaceViewStore()
    opener = WorkspaceViewOpener(views, tmp_path)
    opened = opener.open(view_link("diff", left="A.py", right="B.py", line=2, side="left"))
    selected = opener.open(view_link("diff", left="A.py", right="B.py", line=3))
    assert opened["selection"] == {"document": 0, "line": 2}
    assert selected["id"] == opened["id"]
    assert selected["title"] == "A.py <-> B.py"
    assert selected["selection"] == {"document": 1, "line": 3}


@pytest.mark.parametrize("href", [
    "A.py:0", "A.py:-1", "A.py:9", ":1", "A.py#L0", "A.py:1#L2",
    "A.py#L1.5", "A.py#R1",
    "diff:A.py..A.py#M1", "diff:A.py..A.py#R0", "diff:A.py..A.py#L-1",
])
def test_invalid_line_selection_does_not_open_a_tab(tmp_path, href):
    (tmp_path / "A.py").write_text("first\nsecond", encoding="utf-8")
    views = WorkspaceViewStore()
    with pytest.raises(ValueError):
        WorkspaceViewOpener(views, tmp_path).open(href)
    assert views.list_views() == []


def test_images_reject_line_selection(tmp_path):
    Image.new("RGB", (2, 2)).save(str(tmp_path / "image.png"))
    views = WorkspaceViewStore()
    with pytest.raises(ValueError, match="existing text file"):
        WorkspaceViewOpener(views, tmp_path).open("image.png:1")
    assert views.list_views() == []


def test_reveal_counter_survives_close_and_restart(tmp_path):
    for name in ("A.py", "B.py"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    board = tmp_path / "board.html"
    views = WorkspaceViewStore(board.with_suffix(".pycodex-views.json"))
    opener = WorkspaceViewOpener(views, tmp_path)
    for href in ("A.py", "B.py", "A.py"):
        opener.open(href)
    first, second = views.list_views()
    assert first["reveal_revision"] > second["reveal_revision"]
    views.close(first["id"])
    restored = WorkspaceViewStore(board.with_suffix(".pycodex-views.json"))
    remaining = WorkspaceViewOpener(restored, tmp_path).open("B.py")
    assert remaining["id"] == second["id"]
    assert remaining["reveal_revision"] > first["reveal_revision"]
    assert remaining["revision"] == second["revision"]


def test_concurrent_opens_create_one_tab(tmp_path, monkeypatch):
    (tmp_path / "A.py").write_text("body\n", encoding="utf-8")
    views = WorkspaceViewStore()
    opener = WorkspaceViewOpener(views, tmp_path)
    barrier = threading.Barrier(2)
    read_reference = opener._read_reference

    def read(*args):
        barrier.wait(timeout=5)
        return read_reference(*args)

    monkeypatch.setattr(opener, "_read_reference", read)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(opener.open, ("A.py", "./A.py")))
    assert results[0]["id"] == results[1]["id"]
    assert len(views.list_views()) == 1
    assert views.list_views()[0]["reveal_revision"] == 2


def test_diff_preserves_both_files_and_newlines(tmp_path):
    (tmp_path / "left.py").write_bytes(b"same\r\nold\r\n")
    (tmp_path / "right.py").write_bytes(b"same\r\nnew")
    views = WorkspaceViewStore()
    view = WorkspaceViewOpener(views, tmp_path).open(
        view_link("diff", left="left.py", right="right.py")
    )
    assert view["kind"] == "diff"
    assert [item["text"] for item in view["documents"]] == ["same\r\nold\r\n", "same\r\nnew"]
    (tmp_path / "left.py").write_text("changed left\n", encoding="utf-8")
    (tmp_path / "right.py").unlink()
    changed = views.get(view["id"])
    assert [item["text"] for item in changed["documents"]] == ["changed left\n", ""]
    assert changed["documents"][1]["missing"]
    (tmp_path / "right.py").write_text("back\n", encoding="utf-8")
    assert views.get(view["id"])["documents"][1]["text"] == "back\n"
    assert "missing" not in views.get(view["id"])["documents"][1]


def test_open_tabs_keep_their_order_after_close_and_restore(tmp_path):
    for name in ("A.py", "B.py"):
        (tmp_path / name).write_text("content\n", encoding="utf-8")
    board = tmp_path / "board.html"
    views = WorkspaceViewStore(board.with_suffix(".pycodex-views.json"))
    opener = WorkspaceViewOpener(views, tmp_path)
    file_id = opener.open("A.py")["id"]
    opener.open(view_link("diff", left="A.py", right="B.py"))
    views.close(file_id)
    restored = WorkspaceViewStore(board.with_suffix(".pycodex-views.json"))
    WorkspaceViewOpener(restored, tmp_path).open("B.py")
    assert [view["title"] for view in restored.list_views()] == ["A.py <-> B.py", "B.py"]


def test_images_update_without_serving_the_path(tmp_path):
    image = tmp_path / "plot.png"
    Image.new("RGB", (4, 4), "red").save(str(image))
    views = WorkspaceViewStore()
    first = WorkspaceViewOpener(views, tmp_path).open("plot.png")
    Image.new("RGB", (4, 4), "blue").save(str(image))
    updated = image.read_bytes()
    document = views.get(first["id"])["documents"][0]
    assert views.list_views()[0]["revision"] > first["revision"]
    image.unlink()
    prefix, data = document["image_url"].split(",", 1)
    assert prefix == "data:image/png;base64"
    assert base64.b64decode(data) == updated
    assert "error" in views.get(first["id"])["documents"][0]


def test_view_refresh_detects_atomic_saves_and_skips_unchanged_reads(tmp_path, monkeypatch):
    source = tmp_path / "A.py"
    source.write_text("old\n", encoding="utf-8")
    views = WorkspaceViewStore(tmp_path / "board.pycodex-views.json")
    first = WorkspaceViewOpener(views, tmp_path).open("A.py")
    read_document = views.read_document
    reads = []

    def read(*args, **kwargs):
        reads.append(args[0])
        return read_document(*args, **kwargs)

    monkeypatch.setattr(views, "read_document", read)
    persisted = views.path.read_bytes()
    views.list_views()
    views.get(first["id"])
    assert not reads
    assert views.path.read_bytes() == persisted
    previous_stat = source.stat()
    replacement = tmp_path / "replacement.py"
    replacement.write_text("new\n", encoding="utf-8")
    os.utime(str(replacement), ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
    replacement.replace(source)
    changed = views.get(first["id"])
    assert changed["documents"][0]["text"] == "new\n"
    assert changed["revision"] > first["revision"]
    assert len(reads) == 1
    source.write_text("new\n", encoding="utf-8")
    assert views.get(first["id"])["revision"] == changed["revision"]


def test_view_read_failures_are_visible_and_recover(tmp_path, monkeypatch):
    source = tmp_path / "A.py"
    source.write_text("readable\n", encoding="utf-8")
    views = WorkspaceViewStore()
    view_id = WorkspaceViewOpener(views, tmp_path).open("A.py")["id"]
    source.write_bytes(b"binary\0")
    document = views.get(view_id)["documents"][0]
    assert document["text"] == "readable\n"
    assert "binary" in document["error"]
    source.write_text("recovered\n", encoding="utf-8")
    read_document = views.read_document

    def unavailable(*args, **kwargs):
        raise OSError("temporary read failure")

    monkeypatch.setattr(views, "read_document", unavailable)
    assert views.get(view_id)["documents"][0]["error"] == "temporary read failure"
    monkeypatch.setattr(views, "read_document", read_document)
    document = views.get(view_id)["documents"][0]
    assert document["text"] == "recovered\n" and "error" not in document


def test_earlier_snapshots_without_sources_remain_fixed(tmp_path):
    board = tmp_path / "board.html"
    (tmp_path / "A.py").write_text("new file\n", encoding="utf-8")
    snapshot = {
        "id": "earlier", "kind": "file", "title": "A.py", "created_at": "2026-10-10",
        "documents": [{"label": "A.py", "text": "previous snapshot\n"}],
    }
    board.with_suffix(".pycodex-views.json").write_text(
        json.dumps({"version": 1, "views": [snapshot]}), encoding="utf-8"
    )
    views = WorkspaceViewStore(board.with_suffix(".pycodex-views.json"))
    assert not views.list_views()[0]["live"]
    assert views.get("earlier")["documents"][0]["text"] == "previous snapshot\n"


@pytest.mark.parametrize("href", [
    None, 42, "", "x" * 8193,
    "https://example.test/A.py:42", "mailto:42", "javascript:42", "data:text/html,test",
    "//example.test/A.py", "#L42", "A.py#section", "A.py?download=1",
    "workspace://file?path=A.py", "workspace://unknown", "workspace://canvas",
    "git:A.py", "git::A.py", "git:HEAD:", "diff:A.py", "diff:..A.py", "diff:A.py..",
    "diff:A.py..B.py..C.py", "diff:HEAD:A.py..B.py", "diff://host/a..B.py",
    "diff:A.py..B.py#R1#L2", "diff:A.py..B.py?line=1",
    "git:HEAD:%00", "A%00.py", "A.py\n", "A%xx.py", "A%FF.py",
])
def test_invalid_links_do_not_create_tabs(tmp_path, href):
    views = WorkspaceViewStore(tmp_path / "board.pycodex-views.json")
    with pytest.raises(ValueError):
        WorkspaceViewOpener(views, tmp_path).open(href)
    assert views.list_views() == []
    assert not views.path.exists()


def test_path_contents_are_never_executed(tmp_path):
    name = "$(touch marker).py"
    (tmp_path / name).write_text("literal filename\n", encoding="utf-8")
    opened = WorkspaceViewOpener(WorkspaceViewStore(), tmp_path).open(
        view_link("file", path=name)
    )
    assert opened["documents"][0]["text"] == "literal filename\n"
    assert not (tmp_path / "marker").exists()


@pytest.mark.parametrize("kind", ["size", "lines", "binary", "directory"])
def test_unsupported_or_oversized_files_are_rejected(tmp_path, kind):
    source = tmp_path / "source"
    if kind == "directory":
        source.mkdir()
    else:
        source.write_bytes({
            "size": b"x" * (WorkspaceViewStore.max_file_bytes + 1),
            "lines": b"x\n" * (WorkspaceViewStore.max_text_lines + 1),
            "binary": b"binary\0content",
        }[kind])
    views = WorkspaceViewStore()
    with pytest.raises(ValueError):
        WorkspaceViewOpener(views, tmp_path).open("source")
    assert views.list_views() == []


def test_snapshot_storage_failure_keeps_the_previous_state(tmp_path):
    (tmp_path / "A.py").write_text("first\n", encoding="utf-8")
    views = WorkspaceViewStore(tmp_path / "board.pycodex-views.json")
    opener = WorkspaceViewOpener(views, tmp_path)
    opener.open("A.py")
    previous = views.list_views()
    previous_file = views.path.read_bytes()
    views.max_storage_bytes = len(previous_file)
    (tmp_path / "B.py").write_text("other\n", encoding="utf-8")
    with pytest.raises(ValueError, match="close some tabs"):
        opener.open("B.py")
    assert views.list_views() == previous
    assert views.path.read_bytes() == previous_file
    assert len(list(tmp_path.glob("*.pycodex-views.json.*"))) == 0


def test_failed_reveal_preserves_selection_and_storage(tmp_path, monkeypatch):
    (tmp_path / "A.py").write_text("body\n", encoding="utf-8")
    views = WorkspaceViewStore(tmp_path / "board.pycodex-views.json")
    opener = WorkspaceViewOpener(views, tmp_path)
    opener.open("A.py:1")
    previous = views.list_views()
    persisted = views.path.read_bytes()

    def fail(*args):
        raise OSError("disk is full")

    with monkeypatch.context() as patch:
        patch.setattr(views, "_save", fail)
        with pytest.raises(OSError, match="disk is full"):
            opener.open("A.py:2")
    assert views.list_views() == previous
    assert views.path.read_bytes() == persisted
    assert opener.open("A.py:2")["id"] == previous[0]["id"]
    assert views.list_views()[0]["reveal_revision"] == previous[0]["reveal_revision"] + 1
    assert views.list_views()[0]["selection"] == {"document": 0, "line": 2}


def test_live_update_storage_failure_is_reported_and_can_recover(tmp_path):
    board = tmp_path / "board.html"
    source = tmp_path / "A.py"
    source.write_text("first\n", encoding="utf-8")
    (tmp_path / "B.py").write_text("other\n", encoding="utf-8")
    views = WorkspaceViewStore(board.with_suffix(".pycodex-views.json"))
    opener = WorkspaceViewOpener(views, tmp_path)
    opener.open("A.py")
    opener.open("B.py")
    first, second = views.list_views()
    previous_file = views.path.read_bytes()
    views.max_storage_bytes = len(previous_file)
    source.write_text("updated\n" * 25, encoding="utf-8")
    manager = WorkspaceSessionManager(make_session, board.with_suffix(""), views=views)
    with TestClient(create_app(manager, board.with_suffix(""), work_dir=tmp_path)) as browser:
        for path in ("/api/views", "/api/views/" + first["id"]):
            response = browser.get(path)
            assert response.status_code == 400
            assert "close some tabs" in response.json()["detail"]
        assert views.path.read_bytes() == previous_file
        assert browser.delete("/api/views/" + second["id"]).status_code == 200
        response = browser.get("/api/views/" + first["id"])
        assert response.status_code == 200
        assert response.json()["documents"][0]["text"] == "updated\n" * 25


@pytest.mark.parametrize("form", ["objects", "commits", "working_file"])
def test_git_diff_uses_git_references_and_workspace_relative_paths(git_repository, form):
    repo, first, second = git_repository
    (repo / "data" / "A.py").write_text("working = 3\n", encoding="utf-8")
    href = {
        "objects": view_link("git-diff", left="HEAD~1:data/A.py", right="HEAD:data/A.py"),
        "commits": view_link("git-diff", path="./A.py", base=first, head=second),
        "working_file": view_link("git-diff", path="./A.py", base="HEAD"),
    }[form]
    views = WorkspaceViewStore()
    view = WorkspaceViewOpener(views, repo / "data").open(href)
    assert [document["text"] for document in view["documents"]] == (
        ["value = 2\n", "working = 3\n"]
        if form == "working_file" else ["value = 1\n", "value = 2\n"]
    )
    assert (repo / "data" / "A.py").read_text(encoding="utf-8") == "working = 3\n"
    (repo / "data" / "A.py").write_text("working = 4\n", encoding="utf-8")
    git(repo, "add", "data/A.py")
    git(repo, "commit", "-qm", "third")
    refreshed = views.get(view["id"])
    assert refreshed["documents"][0] == view["documents"][0]
    assert refreshed["documents"][1]["text"] == (
        "working = 4\n" if form == "working_file" else "value = 2\n"
    )
    assert views.list_views()[0]["live"] == (form == "working_file")


def test_git_file_links_pin_a_revision_and_select_a_line(git_repository):
    repo, first, second = git_repository
    views = WorkspaceViewStore()
    opener = WorkspaceViewOpener(views, repo / "data")
    view = opener.open(view_link("file", path="./A.py", rev="HEAD~1", line=1))
    assert view["kind"] == "file"
    assert view["selection"] == {"document": 0, "line": 1}
    assert view["documents"][0]["text"] == "value = 1\n"
    assert opener.open(view_link("file", path="data/A.py", rev=first))["id"] == view["id"]
    assert opener.open(view_link("file", path="./A.py", rev=second))["id"] != view["id"]
    assert not any(item["live"] for item in views.list_views())


def test_git_diff_compares_different_paths(git_repository):
    repo, first, second = git_repository
    view = WorkspaceViewOpener(WorkspaceViewStore(), repo).open(view_link(
        "git-diff", left=first + ":old name.py", right=second + ":new name.py"
    ))
    assert [document["text"] for document in view["documents"]] == ["old\n", "old\n"]


def test_git_tabs_keep_full_revisions_and_reuse_equivalent_forms(git_repository):
    repo, first, second = git_repository
    board = repo / "board.html"
    views = WorkspaceViewStore(board.with_suffix(".pycodex-views.json"))
    opened = WorkspaceViewOpener(views, repo).open(view_link(
        "git-diff", left=first + ":data/A.py", right=second + ":data/A.py"
    ))
    title = "{0}:data/A.py <-> {1}:data/A.py".format(first, second)
    assert opened["title"] == title
    opener = WorkspaceViewOpener(views, repo / "data")
    for href in [
        view_link("git-diff", path="./A.py", base="HEAD~1", head="HEAD"),
        view_link("git-diff", left="HEAD~1:./A.py", right="HEAD:./A.py"),
    ]:
        selected = opener.open(href)
        assert selected["id"] == opened["id"]
        assert selected["title"] == title
    restored = WorkspaceViewStore(board.with_suffix(".pycodex-views.json"))
    opener = WorkspaceViewOpener(restored, repo)
    assert opener.open(view_link(
        "git-diff", path="data/A.py", base=first, head=second
    ))["id"] == opened["id"]
    git(repo, "commit", "--allow-empty", "-qm", "same tree")
    third = git(repo, "rev-parse", "HEAD")
    newest = opener.open(view_link("git-diff", left="HEAD~1:data/A.py", right="HEAD:data/A.py"))
    assert "HEAD~1 ({0}):data/A.py".format(second) in newest["title"]
    assert "HEAD ({0}):data/A.py".format(third) in newest["title"]
    assert newest["id"] != opened["id"]
    assert len(restored.list_views()) == 2


def test_git_line_selection_preserves_revision_syntax_and_tab_identity(git_repository):
    repo, first, second = git_repository
    views = WorkspaceViewStore()
    opener = WorkspaceViewOpener(views, repo)
    opened_id = None
    for href, selection in [
        (view_link("git-diff", left="HEAD~1:data/A.py", right="HEAD:data/A.py",
                   line=1, side="left"), {"document": 0, "line": 1}),
        (view_link("git-diff", left="HEAD~1:data/A.py", right="HEAD:data/A.py",
                   line=2), {"document": 1, "line": 2}),
        (view_link("git-diff", path="data/A.py", base="HEAD~1", head="HEAD",
                   line=1), {"document": 1, "line": 1}),
    ]:
        view = opener.open(href)
        if opened_id is None:
            opened_id = view["id"]
        assert view["id"] == opened_id
        assert view["selection"] == selection
        assert len(views.list_views()) == 1
    working = opener.open(view_link("git-diff", path="data/A.py", base="HEAD", line=1))
    assert working["selection"] == {"document": 1, "line": 1}
    assert working["documents"][1]["source"]["working_tree"]
    previous = views.list_views()
    with pytest.raises(ValueError, match="existing text file"):
        opener.open(view_link("git-diff", path="deleted.py", base="HEAD", line=1))
    assert views.list_views() == previous


def test_numeric_git_paths_are_not_mistaken_for_line_numbers(git_repository):
    repo, first, second = git_repository
    (repo / "123").write_text("numeric filename\n", encoding="utf-8")
    git(repo, "add", "123")
    git(repo, "commit", "-qm", "numeric path")
    opener = WorkspaceViewOpener(WorkspaceViewStore(), repo)
    opened = opener.open(view_link("git-diff", left="HEAD:123", right="HEAD:123"))
    selected = opener.open(view_link("git-diff", left="HEAD:123", right="HEAD:123",
                                     line=1, side="left"))
    assert opened["selection"] is None
    assert selected["id"] == opened["id"]
    assert selected["selection"] == {"document": 0, "line": 1}


def test_saved_titles_retain_paths_and_git_references(tmp_path):
    board = tmp_path / "board.html"
    views = [{
        "id": "git", "kind": "diff", "title": "A.py <-> A.py",
        "created_at": "2026-10-10",
        "documents": [
            {"label": "commita:one/A.py", "text": "old\n"},
            {"label": "commitb:two/A.py", "text": "new\n"},
        ],
    }]
    board.with_suffix(".pycodex-views.json").write_text(
        json.dumps({"version": 1, "views": views}), encoding="utf-8"
    )
    restored = WorkspaceViewStore(board.with_suffix(".pycodex-views.json"))
    assert restored.list_views()[0]["title"] == "commita:one/A.py <-> commitb:two/A.py"
    assert not restored.list_views()[0]["live"]


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs extra Windows privileges")
def test_live_symlinks_follow_file_or_git_working_tree_semantics(git_repository):
    repo, first, second = git_repository
    link = repo / "link.py"
    link.symlink_to("data/A.py")
    git(repo, "add", "link.py")
    git(repo, "commit", "-qm", "link")
    views = WorkspaceViewStore()
    opener = WorkspaceViewOpener(views, repo)
    opened = opener.open("link.py")
    diff = opener.open(view_link("git-diff", path="link.py", base="HEAD"))
    (repo / "data" / "A.py").write_text("followed\n", encoding="utf-8")
    assert views.get(opened["id"])["documents"][0]["text"] == "followed\n"
    assert views.get(diff["id"])["documents"][1]["text"] == "data/A.py"
    link.unlink()
    link.symlink_to("new name.py")
    assert views.get(opened["id"])["documents"][0]["text"] == "old\n"
    assert [item["text"] for item in views.get(diff["id"])["documents"]] == [
        "data/A.py", "new name.py",
    ]


@pytest.mark.parametrize("working", [False, True])
def test_git_diff_represents_deleted_files(git_repository, working):
    repo, first, second = git_repository
    views = WorkspaceViewStore()
    parameters = {"path": "deleted.py", "base": first}
    if not working:
        parameters["head"] = second
    view = WorkspaceViewOpener(views, repo).open(view_link("git-diff", **parameters))
    assert view["documents"][0]["text"] == "removed\n"
    assert view["documents"][1]["text"] == ""
    assert view["documents"][1]["missing"]
    (repo / "deleted.py").write_text("restored\n", encoding="utf-8")
    assert views.get(view["id"])["documents"][1]["text"] == ("restored\n" if working else "")


def test_git_object_diff_allows_one_missing_side(git_repository):
    repo, first, second = git_repository
    opener = WorkspaceViewOpener(WorkspaceViewStore(), repo)
    view = opener.open(view_link(
        "git-diff", left="HEAD~1:new name.py", right="HEAD:new name.py", line=1
    ))
    assert view["documents"][0]["missing"]
    assert view["documents"][1]["text"] == "old\n"
    assert view["selection"] == {"document": 1, "line": 1}


def test_git_diff_rejects_invalid_revisions_and_paths(git_repository):
    repo, first, second = git_repository
    views = WorkspaceViewStore()
    opener = WorkspaceViewOpener(views, repo)
    for href in [
        view_link("git-diff", path="data/A.py", base="missing-revision"),
        view_link("git-diff", path="data/A.py", base="--output=marker"),
        view_link("git-diff", path="data/A.py", base="$(touch marker)"),
        view_link("git-diff", path="../outside.py", base="HEAD"),
        view_link("git-diff", path="missing.py", base="HEAD"),
        view_link("git-diff", path="data", base="HEAD"),
        view_link("git-diff", left="HEAD:data", right="HEAD:data/A.py"),
        view_link("git-diff", left="HEAD:../outside.py", right="HEAD:data/A.py"),
        view_link("file", path="missing.py", rev="HEAD"),
    ]:
        with pytest.raises(ValueError):
            opener.open(href)
    assert views.list_views() == []
    assert not (repo / "marker").exists()


@pytest.mark.parametrize("multi", [False, True])
def test_view_routes_are_authenticated_and_workspace_scoped(tmp_path, multi):
    board = tmp_path / "board.html"
    board.write_text("board", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    (project / "A.py").write_text("private source\n", encoding="utf-8")
    manager = WorkspaceSessionManager(make_session, board.with_suffix(""))
    if multi:
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        (other_dir / "A.py").write_text("other source\n", encoding="utf-8")
        other = WorkspaceSessionManager(make_session, tmp_path / "other")
        registry = WorkspaceRegistry([
            WorkspaceEntry(WorkspaceDefinition("first", board.with_suffix(""), project), manager),
            WorkspaceEntry(WorkspaceDefinition("second", tmp_path / "other", other_dir), other),
        ])
        app = create_multi_workspace_app(registry, password="test-password")
        prefix = "/w/first"
    else:
        app = create_app(manager, board.with_suffix(""), password="test-password", work_dir=project)
        prefix = ""
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "plot.png").write_bytes(b"image")
    with TestClient(app) as browser:
        assert browser.get(prefix + "/api/views").status_code == 401
        assert browser.post(prefix + "/api/views/open", json={"href": "A.py"}).status_code == 401
        assert manager.views.list_views() == []
        assert browser.post("/login", json={"password": "test-password"}).status_code == 200
        response = browser.post(prefix + "/api/views/open", json={"href": "A.py#L1"})
        assert response.status_code == 200
        opened = response.json()
        view_id = opened["view"]["id"]
        assert opened["view"]["documents"][0]["text"] == "private source\n"
        assert opened["view"]["selection"] == {"document": 0, "line": 1}
        assert opened["views"][0]["id"] == view_id
        assert response.headers["cache-control"] == "no-store"
        selected = browser.post(prefix + "/api/views/open", json={"href": str(project / "A.py")})
        assert selected.json()["view"]["id"] == view_id
        relative = browser.post(prefix + "/api/views/open", json={
            "href": "project/A.py#L1", "src_path": str(tmp_path / "guide.md"),
        })
        assert relative.status_code == 200
        assert relative.json()["view"]["id"] == view_id
        assert relative.json()["view"]["selection"] == {"document": 0, "line": 1}
        listed = browser.get(prefix + "/api/views").json()
        assert "private source" not in json.dumps(listed)
        assert browser.get(prefix + "/api/views/" + view_id).json()["documents"][0]["text"] == "private source\n"
        assert browser.get(prefix + "/api/views/not-a-view").status_code == 404
        for payload in [{}, {"href": "missing.py"}, {"href": "A.py:999"}, {"href": 1}]:
            assert browser.post(prefix + "/api/views/open", json=payload).status_code == 400
        for name, media_type in [
            ("content_views.mjs", "text/javascript"), ("content_views.css", "text/css"),
            ("editor.mjs", "text/javascript"), ("editor.css", "text/css"),
        ]:
            asset = browser.get(prefix + "/assets/" + name)
            assert asset.status_code == 200
            assert asset.headers["content-type"].startswith(media_type)
            assert asset.headers["x-content-type-options"] == "nosniff"
            assert asset.headers["content-encoding"] == "gzip"
            assert asset.headers["vary"] == "Accept-Encoding"
        licenses = browser.get(prefix + "/assets/content_views.LICENSE.txt")
        assert licenses.status_code == 200 and "monaco-editor" in licenses.text
        assert browser.get(prefix + "/assets/plot.png").content == b"image"
        assert browser.get(prefix + "/assets/app.py").status_code == 404
        if multi:
            assert browser.get("/w/missing/assets/editor.mjs").status_code == 404
            assert browser.get("/w/second/api/views/" + view_id).status_code == 404
            assert browser.delete("/w/second/api/views/" + view_id).status_code == 404
            other_view = browser.post("/w/second/api/views/open", json={"href": "A.py"}).json()["view"]
            assert other_view["documents"][0]["text"] == "other source\n"
            assert other_view["id"] != view_id
            assert browser.post("/w/missing/api/views/open", json={"href": "A.py"}).status_code == 404
        assert browser.delete(prefix + "/api/views/" + view_id).json() == {"ok": True}
        assert browser.get(prefix + "/api/views").json() == {"views": []}
        assert browser.get(prefix + "/api/views/" + view_id).status_code == 404
        assert browser.delete(prefix + "/api/views/" + view_id).status_code == 404


@pytest.mark.parametrize("multi", [False, True])
def test_opening_view_does_not_block_session_polling(tmp_path, monkeypatch, multi):
    (tmp_path / "A.py").write_text("value = 1\n", encoding="utf-8")
    manager = WorkspaceSessionManager(make_session, None)
    if multi:
        app = create_multi_workspace_app(WorkspaceRegistry([
            WorkspaceEntry(WorkspaceDefinition("first", tmp_path / "state", tmp_path, False), manager),
        ]))
        prefix = "/w/first"
    else:
        app = create_app(manager, None, work_dir=tmp_path)
        prefix = ""
    entered = threading.Event()
    release = threading.Event()
    original = workspace_app._open_view_response

    def blocked_open(*args):
        entered.set()
        assert release.wait(5), "view operation was not released"
        return original(*args)

    monkeypatch.setattr(workspace_app, "_open_view_response", blocked_open)
    with TestClient(app) as browser, ThreadPoolExecutor(max_workers=2) as executor:
        opened = executor.submit(
            browser.post, prefix + "/api/views/open", json={"href": "A.py"}
        )
        try:
            assert entered.wait(5)
            polled = executor.submit(browser.get, prefix + "/api/session")
            assert polled.result(timeout=2).status_code == 200
        finally:
            release.set()
        assert opened.result(timeout=5).json()["view"]["documents"][0]["text"] == (
            "value = 1\n"
        )


def test_view_asset_compression_cache_and_revalidation(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace_app, "__file__", str(tmp_path / "app.py"))
    source = tmp_path / "editor.mjs"
    original = b"export const source = 'original';\n" * 100
    source.write_bytes(original)
    compressions = []
    compress = workspace_app.gzip.compress

    def tracked_compress(body, compresslevel):
        compressions.append(body)
        return compress(body, compresslevel)

    monkeypatch.setattr(workspace_app.gzip, "compress", tracked_compress)
    app = create_app(make_session, None, work_dir=tmp_path)
    with TestClient(app) as browser:
        path = "/assets/editor.mjs"
        encoded = browser.get(path, headers={"Accept-Encoding": "gzip"})
        assert encoded.content == original
        assert encoded.headers["content-encoding"] == "gzip"
        assert int(encoded.headers["content-length"]) < len(original)
        etag = encoded.headers["etag"]
        head = browser.head(path, headers={"Accept-Encoding": "gzip"})
        assert head.content == b""
        assert head.headers["content-length"] == encoded.headers["content-length"]
        assert head.headers["etag"] == etag
        unchanged = browser.get(path, headers={
            "Accept-Encoding": "gzip", "If-None-Match": '"other", W/' + etag,
        })
        assert unchanged.status_code == 304 and unchanged.content == b""
        assert unchanged.headers["vary"] == "Accept-Encoding"
        assert unchanged.headers["cache-control"] == "no-cache"
        plain = browser.get(path, headers={"Accept-Encoding": "identity"})
        assert plain.content == original
        assert "content-encoding" not in plain.headers
        assert int(plain.headers["content-length"]) == len(original)
        assert plain.headers["etag"] != etag
        assert browser.get(path, headers={
            "Accept-Encoding": "identity", "If-None-Match": etag,
        }).status_code == 200

        for accepted, expected in [
            ("", None),
            ("br", None),
            ("GZIP; q=0.5", "gzip"),
            ("gzip;q=0, *;q=1", None),
            ("*;q=0.5", "gzip"),
            ("gzip;q=0.5, identity;q=1", None),
        ]:
            response = browser.get(path, headers={"Accept-Encoding": accepted})
            assert response.content == original
            assert response.headers.get("content-encoding") == expected
        for accepted in ["gzip;q=0, identity;q=0", "*;q=0"]:
            assert browser.get(path, headers={"Accept-Encoding": accepted}).status_code == 406
        for accepted in ["gzip;q=invalid", "gzip;q=2", "gzip;q=nan"]:
            assert browser.get(path, headers={"Accept-Encoding": accepted}).status_code == 400
        assert compressions == [original]

        updated = b"export const source = 'updated contents';\n" * 100
        source.write_bytes(updated)
        changed = browser.get(path, headers={
            "Accept-Encoding": "gzip", "If-None-Match": etag,
        })
        assert changed.status_code == 200 and changed.content == updated
        assert changed.headers["etag"] != etag
        assert browser.get(path, headers={"Accept-Encoding": "identity"}).content == updated
        assert compressions == [original, updated]
        api = browser.get("/api/views", headers={"Accept-Encoding": "gzip"})
        assert api.status_code == 200 and "content-encoding" not in api.headers

    with TestClient(create_app(make_session, None, work_dir=tmp_path)) as browser:
        assert browser.get(path, headers={"Accept-Encoding": "gzip"}).content == updated
    assert compressions == [original, updated, updated]


@pytest.mark.parametrize("toolset", [[], ["apply_patch"]])
def test_model_links_open_only_on_click_without_tools_or_new_messages(tmp_path, monkeypatch, toolset):
    config = tmp_path / "config.toml"
    config.write_text('model = "test"\n', encoding="utf-8")
    target = tmp_path / "project"
    target.mkdir()
    (target / "A.py").write_text("source body\n", encoding="utf-8")
    reply = "[Source](A.py:1)"
    model = ScriptedModelClient([ModelResponse([AssistantMessage(reply)])])
    monkeypatch.setattr("workspace_server.app.build_model", lambda **kwargs: model)
    args = build_parser().parse_args(["--config", str(config), "--toolset"] + toolset)
    entry = _build_workspace_entry(
        WorkspaceDefinition("first", tmp_path / "board", tmp_path), args
    )
    with TestClient(create_multi_workspace_app(WorkspaceRegistry([entry]))) as browser:
        changed = browser.post(
            "/w/first/api/session/message", json={"prompt": "/cd project"}
        ).json()
        assert changed["snapshot"]["cwd"] == str(target)
        session_id = browser.get("/w/first/api/session").json()["session_id"]
        browser.post("/w/first/api/session/message", json={"prompt": "show A.py"})
        before = wait_snapshot(
            browser,
            lambda state: state["turns"] and state["turns"][-1]["response"] == reply,
            "/w/first/api/session",
        )
        assert browser.get("/w/first/api/views").json() == {"views": []}
        assert [tool.name for tool in model.prompts[0].tools] == toolset
        assert "workspace_display" not in repr(model.prompts[0].input)
        assert "Replies and Canvas HTML can use extended links" in repr(model.prompts[0].input)
        assert "diff:BASE..HEAD:path" in repr(model.prompts[0].input)
        assert "Current workspace Canvas file: " + str(tmp_path / "board.html") in repr(model.prompts[0].input)
        assert "<cwd>" + str(target) + "</cwd>" in repr(model.prompts[0].input)
        opened = browser.post("/w/first/api/views/open", json={
            "href": "A.py:1", "session_id": session_id,
        }).json()
        assert opened["view"]["documents"][0]["text"] == "source body\n"
        assert browser.get("/w/first/api/session").json()["snapshot"] == before
        assert len(model.prompts) == 1
    assert "workspace_display" not in get_tools(toolset=["apply_patch"])
    assert len(WorkspaceViewStore(tmp_path / "board.pycodex-views.json").list_views()) == 1


def test_content_tab_controller():
    script = Path(__file__).with_name("workspace_content_views.mjs")
    result = subprocess.run(
        ["node", str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=30, universal_newlines=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_page_imports_match_shipped_module():
    workspace = Path(__file__).parents[1] / "workspace_server"
    # Canvas must not load Monaco's multi-megabyte code or stylesheet.
    assert (workspace / "content_views.mjs").stat().st_size < 30000
    assert (workspace / "content_views.css").stat().st_size < 5000
    page = (workspace / "workspace.html").read_text(encoding="utf-8")
    imports = re.findall(
        r'^\s*import (\{[^\n]+\}) from "\./assets/content_views\.mjs";',
        page, re.MULTILINE,
    )
    assert imports
    script = "\n".join(
        'import {0} from "./content_views.mjs";'.format(names) for names in imports
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script], cwd=str(workspace),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, universal_newlines=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
