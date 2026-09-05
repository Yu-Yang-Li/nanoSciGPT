import pytest

from nanoscigpt import upstream


def local_origin(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    upstream.git(source, "init")
    (source / "example.txt").write_text("original\n", encoding="utf-8")
    upstream.git(source, "add", "example.txt")
    upstream.git(source, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit", "-m", "fixture")
    pin = upstream.git(source, "rev-parse", "HEAD")
    monkeypatch.setitem(upstream.PROJECTS, "fixture", (str(source), pin))
    return source, pin


def test_checkout_can_resume_an_interrupted_initial_fetch(tmp_path, monkeypatch):
    source, pin = local_origin(tmp_path, monkeypatch)
    destination = tmp_path / "destination"
    destination.mkdir()
    upstream.git(destination, "init")
    upstream.git(destination, "remote", "add", "origin", str(source))
    upstream.checkout("fixture", destination)
    assert upstream.git(destination, "rev-parse", "HEAD") == pin
    assert (destination / "example.txt").read_text() == "original\n"


def test_checkout_does_not_reset_student_edits(tmp_path, monkeypatch):
    _, pin = local_origin(tmp_path, monkeypatch)
    destination = tmp_path / "destination"
    upstream.checkout("fixture", destination)
    (destination / "example.txt").write_text("student edit\n")
    upstream.checkout("fixture", destination)
    assert (destination / "example.txt").read_text() == "student edit\n"
    assert upstream.git(destination, "rev-parse", "HEAD") == pin


def test_assignment_changes_only_the_training_configuration():
    source = 'class Model:\n    def forward(self):\n        device = x.device\n\ndef train():\n    device = "cuda"\n'
    result = upstream.assignments(source, {"device": '"cpu"'}, indent="    ")
    namespace = {}
    exec(compile(result, "experiment.py", "exec"), namespace)
    assert namespace["train"].__code__.co_consts == (None, "cpu")
    assert namespace["Model"].forward.__code__.co_names == ("x", "device")
    with pytest.raises(ValueError):
        upstream.assignments(source, {"missing_setting": "1"})
