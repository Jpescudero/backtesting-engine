"""Regression tests for default data hub configuration."""

from pathlib import Path

from src.config import paths


def test_default_local_hubs_include_lowercase_profile() -> None:
    """Ensure the new lowercase Windows profile is part of the defaults."""

    expected = Path(r"C:/Users/jorge/Market Data")
    assert expected in paths.DEFAULT_LOCAL_HUBS


def test_load_data_hubs_from_file_supports_lowercase_profile(tmp_path: Path) -> None:
    """Ensure config loader preserves the lowercase Windows profile path."""

    config_path = tmp_path / "data_roots.json"
    config_path.write_text(
        """
{
  "local_hubs": [
    "C:/Users/jorge/Market Data"
  ],
  "cloud_hubs": []
}
        """.strip(),
        encoding="utf-8",
    )

    local_hubs, cloud_hubs = paths._load_data_hubs_from_file(config_path)

    assert local_hubs == [Path(r"C:/Users/jorge/Market Data")]
    assert cloud_hubs == []
