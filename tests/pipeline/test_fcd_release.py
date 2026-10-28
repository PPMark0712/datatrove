import zipfile

import pytest

from datatrove.pipeline.fcd.build_dict.package_data import RESOURCE_FILES, package, validate_archive


def test_release_zip_preserves_all_four_resources(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for name in RESOURCE_FILES:
        (data / name).write_bytes(f"UTF-8 中文 {name}\n".encode())
    (data / "manifest.json").write_text("obsolete")
    output = package(data, tmp_path / "dist/fcd_data.zip")
    assert validate_archive(output) == sorted(f"data/{name}" for name in RESOURCE_FILES)
    with zipfile.ZipFile(output) as archive:
        for name in RESOURCE_FILES:
            assert archive.read(f"data/{name}") == (data / name).read_bytes()
        assert "data/manifest.json" not in archive.namelist()


def test_missing_resource_does_not_create_a_release_zip(tmp_path):
    with pytest.raises(ValueError, match="Missing or empty"):
        package(tmp_path, tmp_path / "fcd_data.zip")
    assert not (tmp_path / "fcd_data.zip").exists()


def test_release_validation_rejects_unexpected_paths(tmp_path):
    output = tmp_path / "bad.zip"
    with zipfile.ZipFile(output, "w") as archive:
        for name in RESOURCE_FILES:
            archive.writestr(f"data/{name}", "test")
        archive.writestr("../outside.txt", "unexpected")
    with pytest.raises(ValueError, match="exactly"):
        validate_archive(output)


def test_release_validation_rejects_empty_resources(tmp_path):
    output = tmp_path / "bad.zip"
    with zipfile.ZipFile(output, "w") as archive:
        for name in RESOURCE_FILES:
            archive.writestr(f"data/{name}", "")
    with pytest.raises(ValueError, match="Empty FCD resource"):
        validate_archive(output)
