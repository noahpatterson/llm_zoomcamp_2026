"""Load Pydantic Logfire traces into DuckDB (dataset: agent_traces).

Based on the official dlt Logfire export example:
https://dlthub.com/docs/examples/logfire_telemetry_export
"""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import dlt
from dotenv import load_dotenv
from logfire.query_client import LogfireQueryClient

load_dotenv(Path(__file__).resolve().parent / ".env")


@dlt.resource(
    name="records",
    write_disposition="merge",
    primary_key="span_id",
)
def records(
    read_token: str = dlt.secrets.value,
    batch_size: int = 10_000,
    min_timestamp=dlt.sources.incremental(
        "start_timestamp",
        initial_value=datetime(1970, 1, 1, tzinfo=ZoneInfo("UTC")),
    ),
):
    """Fetch Logfire `records` (spans and logs) incrementally.

    Spans that share a `trace_id` form a trace — the primary Live View table.
    """
    with LogfireQueryClient(read_token=read_token) as client:
        batch = client.query_arrow(
            sql=f"SELECT * FROM records LIMIT {batch_size}",
            min_timestamp=min_timestamp.start_value,
            limit=batch_size,
        )
        yield batch.to_pylist()


@dlt.source
def logfire_source(read_token: str = dlt.secrets.value):
    """Logfire traces → DuckDB dataset `agent_traces`."""
    return records(read_token=read_token)


def load_logfire_traces() -> None:
    import os

    pipeline = dlt.pipeline(
        pipeline_name="logfire",
        destination="duckdb",
        dataset_name="agent_traces",
    )

    token = os.environ.get("LOGFIRE_READ_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing LOGFIRE_READ_TOKEN. Add it to dlt-homework/.env "
            "(or set [sources.logfire_source] read_token in .dlt/secrets.toml)."
        )

    load_info = pipeline.run(logfire_source(read_token=token))
    print(load_info)
    print(pipeline.last_trace.last_normalize_info)


if __name__ == "__main__":
    load_logfire_traces()
