from src.lr0.data import prepare_split, validate_split, load_split

def test_split_sizes_and_integrity(tmp_path, monkeypatch):
    import src.lr0.data as d
    monkeypatch.setattr(d, "SPLIT_PATH", tmp_path / "split.json")
    monkeypatch.setattr(d, "MANIFEST_PATH", tmp_path / "manifest.json")
    prepare_split(0)
    assert validate_split(tmp_path / "split.json")
    s = load_split(tmp_path / "split.json")
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (1077, 360, 360)
