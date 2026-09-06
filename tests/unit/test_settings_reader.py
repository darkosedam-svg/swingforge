"""Tests for `swingforge.adapters.settings_reader.TransientSettingsReader`."""

from __future__ import annotations

from pathlib import Path

from swingforge.adapters.settings_reader import TransientSettingsReader
from swingforge.adapters.store import Store
from swingforge.core.settings import Settings


def test_reads_back_the_defaults_when_nothing_was_ever_written(tmp_path: Path) -> None:
    path = tmp_path / "v.duckdb"
    store = Store(path)
    store.close()

    reader = TransientSettingsReader(path)
    version, settings = reader.current()

    assert version == 0
    assert settings == Settings()


def test_version_bump_from_another_store_instance_is_visible(tmp_path: Path) -> None:
    path = tmp_path / "v.duckdb"
    store = Store(path)
    store.close()

    reader = TransientSettingsReader(path)
    version_before, settings_before = reader.current()
    assert version_before == 0
    assert settings_before.risk_pct == 0.01

    writer = Store(path)
    writer.write_settings(Settings(risk_pct=0.02), actor="test")
    writer.close()

    version_after, settings_after = reader.current()
    assert version_after == version_before + 1
    assert settings_after.risk_pct == 0.02


def test_each_call_opens_and_closes_its_own_connection(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "v.duckdb"
    store = Store(path)
    store.close()

    opens = []
    closes = []
    real_init = Store.__init__
    real_close = Store.close

    def counting_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        opens.append(1)
        real_init(self, *args, **kwargs)

    def counting_close(self):  # type: ignore[no-untyped-def]
        closes.append(1)
        real_close(self)

    monkeypatch.setattr(Store, "__init__", counting_init)
    monkeypatch.setattr(Store, "close", counting_close)

    reader = TransientSettingsReader(path)
    reader.current()
    reader.current()

    assert len(opens) == 2
    assert len(closes) == 2
