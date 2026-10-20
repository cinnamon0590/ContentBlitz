"""The brand profile file: a missing or unreadable file falls back to the default profile."""

from src.web_app.components import BRAND_FILE_ENV, brand_file, load_saved_brand
from src.workflow.state_management import BrandProfile


def test_brand_file_location(monkeypatch, tmp_path):
    monkeypatch.delenv(BRAND_FILE_ENV, raising=False)
    assert brand_file().parts[-2:] == ("outputs", "brand_profile.json")
    monkeypatch.setenv(BRAND_FILE_ENV, str(tmp_path / "b.json"))
    assert brand_file() == tmp_path / "b.json"


def test_load_saved_brand_falls_back_to_default(monkeypatch, tmp_path):
    path = tmp_path / "b.json"
    monkeypatch.setenv(BRAND_FILE_ENV, str(path))
    assert load_saved_brand() == BrandProfile()  # no file yet
    path.write_text("{not json")
    assert load_saved_brand() == BrandProfile()  # unreadable
    path.write_text(BrandProfile(name="Crumb & Co", tone="warm").model_dump_json())
    assert load_saved_brand().name == "Crumb & Co"
