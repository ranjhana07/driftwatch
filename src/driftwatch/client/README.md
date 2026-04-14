# client

The only part of driftwatch that runs inside your serving process. It depends on `httpx` and `pydantic` and nothing else; install the base package without extras.

```python
from driftwatch import Client

monitor = Client("http://collector:8000", model="churn", model_version="2024-06", api_key="...")

with monitor.measure() as timer:
    score = model.predict_proba(row)[0, 1]

event_id = monitor.log_prediction(
    features=row_dict,
    prediction=int(score > 0.5),
    score=float(score),
    embedding=embedding.tolist(),
    latency_ms=timer.ms,
)

monitor.log_label(event_id, label=1)
```

## Behaviour

- `log_prediction` and `log_label` put a dict on a bounded in-memory queue and return. They never wait on the network. When the queue is full the event is dropped and counted in `client.stats.dropped_queue_full`.
- A background thread sends batches of up to `batch_size` events, or whatever is queued after `flush_interval` seconds.
- 5xx responses, 408, 429 and connection errors are retried with exponential backoff (`max_retries`, `backoff_base`). Other 4xx responses are counted as rejected and not retried.
- With `buffer_dir` set, batches that still fail are written to disk as JSON files and replayed, oldest first, once the collector answers again. The buffer is capped by `buffer_max_bytes`; the oldest files go first.
- `sample_rate` keeps a deterministic share of events, chosen by hashing the event id. A label for a sampled-out event is skipped too, so the collector never sees labels without predictions.
- `flush()` blocks until everything queued has been delivered or spilled. `close()` flushes and stops the thread, and is also registered with `atexit`.

`AdminClient` in `driftwatch.client.admin` wraps the model registration, reference upload and monitor endpoints.
