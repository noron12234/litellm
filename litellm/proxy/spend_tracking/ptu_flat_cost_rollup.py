"""
Daily rollup for per-model PTU (provisioned throughput) flat cost.

v1 reads PTU config straight off the model deployment
(``LiteLLM_ProxyModelTable.model_info``): a deployment carrying ``ptu_count``
and ``cost_per_ptu_per_hour`` accrues flat cost of
``ptu_count * cost_per_ptu_per_hour * active_hours`` for a given UTC day, where
``active_hours`` is the overlap between the day and the optional
``[ptu_effective_from, ptu_effective_to)`` window (a window opening at 23:00
charges one hour that day). The amount is written to ``LiteLLM_DailyTeamSpend``
under a sentinel api_key so the rows are distinguishable from per-request rows
and share the existing unique constraint.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any  # noqa: TID251  # prisma model rows and client are dynamically typed

from litellm._logging import verbose_proxy_logger
from litellm.constants import PTU_ROLLUP_JOB_ID, PTU_SENTINEL_API_KEY

_HOURS_PER_DAY = 24


@dataclass(frozen=True, slots=True)
class RollupResult:
    day: date
    models_processed: int
    rows_written: int


@dataclass(frozen=True, slots=True)
class PTUModel:
    """A model deployment carrying valid manual PTU config."""

    model_id: str
    model_name: str
    team_id: str
    ptu_count: int
    cost_per_ptu_per_hour: float
    effective_from: datetime | None = None
    effective_to: datetime | None = None


def _parse_utc_datetime(value: object) -> datetime | None:
    """Parse a model_info datetime (ISO string or datetime) into a UTC-aware datetime, else None."""
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_ptu_model(row: Any) -> PTUModel | None:  # noqa: ANN401  # prisma model row is dynamically typed
    """Return a PTUModel when the deployment carries valid manual PTU config, else None.

    Valid means model_info has a positive ptu_count, a non-negative
    cost_per_ptu_per_hour, and a team_id (1 model -> 1 team).
    """
    model_info = getattr(row, "model_info", None)
    if not isinstance(model_info, dict):
        return None
    ptu_count = model_info.get("ptu_count")
    cost_per_hour = model_info.get("cost_per_ptu_per_hour")
    team_id = model_info.get("team_id")
    if ptu_count is None or cost_per_hour is None or not team_id:
        return None
    try:
        ptu_count_int = int(ptu_count)
        cost_per_hour_float = float(cost_per_hour)
    except (TypeError, ValueError):
        return None
    if ptu_count_int <= 0 or cost_per_hour_float < 0:
        return None
    return PTUModel(
        model_id=str(getattr(row, "model_id", "") or ""),
        model_name=str(getattr(row, "model_name", "") or ""),
        team_id=str(team_id),
        ptu_count=ptu_count_int,
        cost_per_ptu_per_hour=cost_per_hour_float,
        effective_from=_parse_utc_datetime(model_info.get("ptu_effective_from")),
        effective_to=_parse_utc_datetime(model_info.get("ptu_effective_to")),
    )


def _active_hours_on_day(model: PTUModel, day: date) -> float:
    """Hours the model's PTU window overlaps ``day`` (UTC), clamped to [0, 24]."""
    day_start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)
    start = max(day_start, model.effective_from) if model.effective_from else day_start
    end = min(day_end, model.effective_to) if model.effective_to else day_end
    if end <= start:
        return 0.0
    return (end - start).total_seconds() / 3600.0


def _compute_daily_flat_cost(model: PTUModel, day: date) -> float:
    """Flat cost for ``day``: ptu_count * cost_per_ptu_per_hour * active_hours."""
    return float(model.ptu_count) * model.cost_per_ptu_per_hour * _active_hours_on_day(model, day)


async def _upsert_ptu_daily_row(
    prisma_client: Any,  # noqa: ANN401  # prisma client is dynamically typed
    *,
    team_id: str,
    model: str,
    date_str: str,
    source_model_id: str,
    flat_cost: float,
) -> None:
    """Idempotent upsert of a sentinel-api_key row on LiteLLM_DailyTeamSpend."""
    where = {  # mutable-ok: prisma upsert filter payload
        "team_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint": {  # mutable-ok: prisma composite-key filter
            "team_id": team_id,
            "date": date_str,
            "api_key": PTU_SENTINEL_API_KEY,
            "model": model,
            "custom_llm_provider": "",
            "mcp_namespaced_tool_name": "",
            "endpoint": "",
        }
    }
    now = datetime.now(timezone.utc)
    await prisma_client.db.litellm_dailyteamspend.upsert(
        where=where,
        data={  # mutable-ok: prisma upsert data payload
            "create": {  # mutable-ok: prisma create payload
                "team_id": team_id,
                "date": date_str,
                "api_key": PTU_SENTINEL_API_KEY,
                "model": model,
                "custom_llm_provider": "",
                "mcp_namespaced_tool_name": "",
                "endpoint": "",
                "ptu_flat_cost": flat_cost,
                "ptu_source_model_id": source_model_id,
            },
            "update": {  # mutable-ok: prisma update payload
                "ptu_flat_cost": flat_cost,
                "ptu_source_model_id": source_model_id,
                "updated_at": now,
            },
        },
    )


async def run_ptu_flat_cost_rollup(
    prisma_client: Any,  # noqa: ANN401  # prisma client is dynamically typed
    target_date: date | None = None,
) -> RollupResult:
    """Rollup one UTC day of flat PTU cost across all PTU-configured model deployments.

    Defaults to yesterday UTC. Idempotent under the LiteLLM_DailyTeamSpend unique
    constraint on every invocation path.
    """
    day = target_date or (datetime.now(timezone.utc).date() - timedelta(days=1))

    if prisma_client is None:
        verbose_proxy_logger.warning("PTU rollup: prisma_client is None, skipping")
        return RollupResult(day=day, models_processed=0, rows_written=0)

    date_str = day.isoformat()

    rows = await prisma_client.db.litellm_proxymodeltable.find_many()
    ptu_models = tuple(parsed for parsed in (_parse_ptu_model(row) for row in rows) if parsed is not None)

    rows_written = 0
    for ptu_model in ptu_models:
        flat_cost = _compute_daily_flat_cost(ptu_model, day)
        if flat_cost <= 0:
            continue
        try:
            await _upsert_ptu_daily_row(
                prisma_client,
                team_id=ptu_model.team_id,
                model=ptu_model.model_name,
                date_str=date_str,
                source_model_id=ptu_model.model_id,
                flat_cost=flat_cost,
            )
            rows_written += 1
        except Exception as exc:  # noqa: BLE001  # one bad model must not stop the batch; logged and continued
            verbose_proxy_logger.error(
                "PTU rollup: upsert failed for model=%s day=%s: %s",
                ptu_model.model_id,
                date_str,
                exc,
            )

    verbose_proxy_logger.info(
        "PTU rollup for %s: %d PTU models processed, %d rows written",
        date_str,
        len(ptu_models),
        rows_written,
    )
    return RollupResult(day=day, models_processed=len(ptu_models), rows_written=rows_written)


__all__ = (
    "PTU_ROLLUP_JOB_ID",
    "PTU_SENTINEL_API_KEY",
    "PTUModel",
    "RollupResult",
    "run_ptu_flat_cost_rollup",
)
