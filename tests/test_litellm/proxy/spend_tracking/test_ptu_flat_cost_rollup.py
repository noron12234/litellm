"""Tests for the per-model PTU flat-cost daily rollup."""

import types
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.constants import PTU_SENTINEL_API_KEY
from litellm.proxy.spend_tracking.ptu_flat_cost_rollup import (
    PTUModel,
    _active_hours_on_day,
    _compute_daily_flat_cost,
    _parse_ptu_model,
    run_ptu_flat_cost_rollup,
)

DAY = date(2026, 7, 30)


def _model_row(model_id="m1", model_name="gpt-4o-mini-ptu", model_info=None):
    row = MagicMock()
    row.model_id = model_id
    row.model_name = model_name
    row.model_info = model_info
    return row


def _model(**overrides):
    base = dict(model_id="m", model_name="x", team_id="t", ptu_count=5, cost_per_ptu_per_hour=2.0)
    base.update(overrides)
    return PTUModel(**base)


def test_full_day_when_no_window():
    # 5 PTU * $2.00/hr * 24h = $240
    assert _compute_daily_flat_cost(_model(), DAY) == pytest.approx(240.0)


def test_window_opening_at_2300_charges_one_hour():
    m = _model(effective_from=datetime(2026, 7, 30, 23, 0, tzinfo=timezone.utc))
    assert _active_hours_on_day(m, DAY) == pytest.approx(1.0)
    # 5 * 2.0 * 1 = 10
    assert _compute_daily_flat_cost(m, DAY) == pytest.approx(10.0)


def test_window_closing_at_0600_charges_six_hours():
    m = _model(effective_to=datetime(2026, 7, 30, 6, 0, tzinfo=timezone.utc))
    assert _active_hours_on_day(m, DAY) == pytest.approx(6.0)
    assert _compute_daily_flat_cost(m, DAY) == pytest.approx(60.0)


def test_window_fully_covering_day_charges_24h():
    m = _model(
        effective_from=datetime(2026, 7, 1, tzinfo=timezone.utc),
        effective_to=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )
    assert _active_hours_on_day(m, DAY) == pytest.approx(24.0)


def test_window_before_day_charges_zero():
    m = _model(effective_to=datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc))
    assert _active_hours_on_day(m, DAY) == 0.0
    assert _compute_daily_flat_cost(m, DAY) == 0.0


def test_window_after_day_charges_zero():
    m = _model(effective_from=datetime(2026, 7, 31, 1, 0, tzinfo=timezone.utc))
    assert _active_hours_on_day(m, DAY) == 0.0


def test_naive_effective_from_is_treated_as_utc():
    parsed = _parse_ptu_model(
        _model_row(
            model_info={
                "ptu_count": 5,
                "cost_per_ptu_per_hour": 2.0,
                "team_id": "t",
                "ptu_effective_from": "2026-07-30T23:00:00",
            }
        )
    )
    assert parsed is not None
    assert _active_hours_on_day(parsed, DAY) == pytest.approx(1.0)


def test_effective_from_with_z_suffix_parses():
    parsed = _parse_ptu_model(
        _model_row(
            model_info={
                "ptu_count": 1,
                "cost_per_ptu_per_hour": 1.0,
                "team_id": "t",
                "ptu_effective_from": "2026-07-30T18:00:00Z",
            }
        )
    )
    assert parsed is not None
    assert _active_hours_on_day(parsed, DAY) == pytest.approx(6.0)


@pytest.mark.parametrize(
    "model_info",
    [
        None,
        {},
        {"ptu_count": 5},
        {"cost_per_ptu_per_hour": 2.0},
        {"ptu_count": 5, "cost_per_ptu_per_hour": 2.0},  # missing team_id
        {"ptu_count": 0, "cost_per_ptu_per_hour": 2.0, "team_id": "t"},
        {"ptu_count": 5, "cost_per_ptu_per_hour": -1.0, "team_id": "t"},
        {"ptu_count": "not-int", "cost_per_ptu_per_hour": 2.0, "team_id": "t"},
    ],
)
def test_parse_ptu_model_rejects_invalid(model_info):
    assert _parse_ptu_model(_model_row(model_info=model_info)) is None


def _prisma_with_models(rows):
    prisma = MagicMock()
    table = MagicMock()
    table.find_many = AsyncMock(return_value=rows)
    table.upsert = AsyncMock()
    prisma.db = types.SimpleNamespace(litellm_proxymodeltable=table, litellm_dailyteamspend=table)
    return prisma, table


@pytest.mark.asyncio
async def test_rollup_writes_sentinel_row_with_hourly_cost():
    rows = [_model_row(model_info={"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "team_id": "team_x"})]
    prisma, table = _prisma_with_models(rows)

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.models_processed == 1
    assert result.rows_written == 1
    created = table.upsert.await_args.kwargs["data"]["create"]
    assert created["api_key"] == PTU_SENTINEL_API_KEY
    assert created["ptu_flat_cost"] == pytest.approx(240.0)
    assert created["team_id"] == "team_x"
    assert created["model"] == "gpt-4o-mini-ptu"


@pytest.mark.asyncio
async def test_rollup_skips_zero_active_hours():
    rows = [
        _model_row(
            model_info={
                "ptu_count": 5,
                "cost_per_ptu_per_hour": 2.0,
                "team_id": "team_x",
                "ptu_effective_from": "2026-08-01T00:00:00Z",
            }
        )
    ]
    prisma, table = _prisma_with_models(rows)

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.models_processed == 1
    assert result.rows_written == 0
    table.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_rollup_skips_models_without_ptu_config():
    rows = [
        _model_row(model_id="plain", model_info={"team_id": "team_x"}),
        _model_row(model_id="ptu", model_info={"ptu_count": 3, "cost_per_ptu_per_hour": 1.0, "team_id": "team_y"}),
    ]
    prisma, table = _prisma_with_models(rows)

    result = await run_ptu_flat_cost_rollup(prisma, target_date=DAY)

    assert result.models_processed == 1
    assert result.rows_written == 1
