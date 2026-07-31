"""Cluster-shared cache of the `LiteLLM_ProxyModelTable` rows.

Every pod serves `/model/info` and routes from its own in-process `llm_router`, so model
writes are invisible to sibling pods until they re-read the table. This puts that read
behind a `DualCache` (in-memory + Redis, the same shape as `litellm_config_cache`) and
exposes the invalidation every model write path calls.

Cached entries are the raw DB payload: `litellm_params` values stay encrypted exactly as
stored, and each read validates a fresh copy so callers that decrypt in place cannot
corrupt the cache. The Redis TTL is the config reload interval, so a write path that
forgets to invalidate is never staler than the DB poll it replaced.
"""

import hashlib
from typing import Final, Mapping, Sequence

from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS
from litellm.models.model import LiteLLM_ProxyModelTable

MODEL_LIST_CACHE_KEY: Final[str] = "litellm_proxy:model_list"
MODEL_LIST_CACHE_IN_MEMORY_TTL_SECONDS: Final[float] = 1.0
MODEL_LIST_CACHE_REDIS_TTL_SECONDS: Final[float] = float(PROXY_CONFIG_RELOAD_INTERVAL_SECONDS)

model_list_cache: DualCache = DualCache(
    default_in_memory_ttl=MODEL_LIST_CACHE_IN_MEMORY_TTL_SECONDS,
    default_redis_ttl=MODEL_LIST_CACHE_REDIS_TTL_SECONDS,
)

_MODEL_ROWS_ADAPTER: Final[TypeAdapter[tuple[LiteLLM_ProxyModelTable, ...]]] = TypeAdapter(
    tuple[LiteLLM_ProxyModelTable, ...]
)


def _row_as_mapping(row: object) -> Mapping[str, object]:
    if isinstance(row, BaseModel):
        return row.model_dump()
    if isinstance(row, Mapping):
        return row
    raise TypeError(f"Unsupported model row type: {type(row).__name__}")


def parse_model_rows(rows: Sequence[object]) -> tuple[LiteLLM_ProxyModelTable, ...]:
    return _MODEL_ROWS_ADAPTER.validate_python(tuple(_row_as_mapping(row) for row in rows))


def model_rows_fingerprint(models: Sequence[LiteLLM_ProxyModelTable]) -> str:
    """Identity of a model list for change detection; `updated_at` moves on every row write."""
    parts = sorted(f"{model.model_id}@{model.updated_at.isoformat() if model.updated_at else ''}" for model in models)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


async def get_cached_model_rows(cache: DualCache = model_list_cache) -> tuple[LiteLLM_ProxyModelTable, ...] | None:
    """Cached model rows, or None when the cache is empty or unreadable."""
    cached = await cache.async_get_cache(MODEL_LIST_CACHE_KEY)
    if cached is None:
        return None
    try:
        return _MODEL_ROWS_ADAPTER.validate_python(cached)
    except ValidationError as e:
        verbose_proxy_logger.warning("model_list_cache: discarding unreadable cache entry - %s", str(e))
        return None


async def set_cached_model_rows(
    models: Sequence[LiteLLM_ProxyModelTable],
    cache: DualCache = model_list_cache,
) -> None:
    """Populate both tiers, each with its own TTL.

    `DualCache.async_set_cache` writes one TTL to both tiers, which cannot express what this
    cache needs: the in-memory copy has to lapse fast enough for a sibling pod's invalidation
    to be seen, while the shared copy has to outlive it or every pod re-reads the DB on every
    sync.
    """
    payload = tuple(model.model_dump(mode="json") for model in models)
    await cache.in_memory_cache.async_set_cache(
        MODEL_LIST_CACHE_KEY,
        payload,
        ttl=MODEL_LIST_CACHE_IN_MEMORY_TTL_SECONDS,
    )
    if cache.redis_cache is not None:
        await cache.redis_cache.async_set_cache(
            MODEL_LIST_CACHE_KEY,
            payload,
            ttl=MODEL_LIST_CACHE_REDIS_TTL_SECONDS,
        )


async def invalidate_model_list_cache(cache: DualCache = model_list_cache) -> None:
    """Evict from both cache layers; call after every model table write."""
    await cache.async_delete_cache(MODEL_LIST_CACHE_KEY)
