"""Contract tests for Settings and the SettingsReader protocol."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from swingforge.core.settings import (
    InMemorySettingsReader,
    Settings,
    SettingsReader,
    StaticSettingsReader,
)


def test_defaults() -> None:
    s = Settings()
    assert s.risk_pct == 0.01
    assert s.enabled_instruments == {}
    assert s.paper_configs == ()
    assert s.session == "none"
    assert s.time_stop_bars == 10
    assert s.kill_switch is False


def test_risk_pct_is_capped_at_two_percent() -> None:
    assert Settings(risk_pct=0.02).risk_pct == 0.02
    with pytest.raises(ValidationError):
        Settings(risk_pct=0.03)


def test_risk_pct_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(risk_pct=0.0)
    with pytest.raises(ValidationError):
        Settings(risk_pct=-0.01)


def test_time_stop_bars_must_be_at_least_one() -> None:
    assert Settings(time_stop_bars=1).time_stop_bars == 1
    with pytest.raises(ValidationError):
        Settings(time_stop_bars=0)


def test_session_is_one_of_three() -> None:
    for session in ("none", "london_ny", "active"):
        assert Settings(session=session).session == session  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Settings(session="asia")  # type: ignore[arg-type]


def test_enabled_instruments_and_paper_configs() -> None:
    s = Settings(
        enabled_instruments={"hyperliquid": ("BTC", "ETH"), "oanda": ("EUR_USD",)},
        paper_configs=("ict.fixed_r_2.none.BTC",),
    )
    assert s.enabled_instruments["hyperliquid"] == ("BTC", "ETH")
    assert s.paper_configs == ("ict.fixed_r_2.none.BTC",)


def test_settings_are_frozen() -> None:
    s = Settings()
    with pytest.raises(ValidationError):
        s.kill_switch = True  # type: ignore[misc]


def test_static_reader_returns_a_stable_version() -> None:
    settings = Settings(risk_pct=0.015, kill_switch=True)
    reader = StaticSettingsReader(settings)

    version, current = reader.current()
    assert version == 0
    assert current == settings

    again_version, again = reader.current()
    assert again_version == version
    assert again == settings


def test_static_reader_defaults_to_default_settings() -> None:
    assert StaticSettingsReader().current() == (0, Settings())


def test_static_reader_satisfies_the_protocol() -> None:
    assert isinstance(StaticSettingsReader(), SettingsReader)


def test_static_reader_exposes_its_settings_read_only() -> None:
    settings = Settings(risk_pct=0.015)
    reader = StaticSettingsReader(settings)
    assert reader.settings == settings
    with pytest.raises(AttributeError):
        reader.settings = Settings()  # type: ignore[misc]
    assert reader.current() == (0, settings)


def test_in_memory_reader_starts_at_version_zero() -> None:
    assert InMemorySettingsReader().current() == (0, Settings())
    seeded = Settings(risk_pct=0.005)
    assert InMemorySettingsReader(seeded).current() == (0, seeded)


def test_in_memory_reader_bumps_the_version_on_every_set() -> None:
    reader = InMemorySettingsReader()
    updated = Settings(risk_pct=0.02, kill_switch=True)

    reader.set(updated)
    assert reader.current() == (1, updated)

    # Even setting the same values again is a change the engine must re-apply.
    reader.set(updated)
    assert reader.current() == (2, updated)

    reader.set(Settings())
    assert reader.current() == (3, Settings())


def test_in_memory_reader_satisfies_the_protocol() -> None:
    assert isinstance(InMemorySettingsReader(), SettingsReader)


def test_settings_rejects_unknown_fields() -> None:
    """A typo such as ``kill_swtich`` must fail loudly, never silently return defaults."""
    import pytest
    from pydantic import ValidationError

    from swingforge.core.settings import Settings

    with pytest.raises(ValidationError):
        Settings(kill_swtich=True)  # type: ignore[call-arg]
