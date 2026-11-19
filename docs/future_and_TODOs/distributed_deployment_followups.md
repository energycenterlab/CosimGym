# Distributed deployment — follow-ups

Open enhancements to distributed SSH federate spawning, deferred when v1 was built
(plan: `docs/changes_reports/distributed_ssh_spawning_plan_DONE.md`; user guide:
`docs/user_guide/distributed_deployment.md`). None is implemented.

1. **MinIO/S3 live results sink**: federates upload parquet batches to MinIO (already in `src/docker-compose.yaml`, creds there) during run → no rsync-back, dashboard reads central store, survives remote-disk loss. Natural extension of `AsyncStorageWriter`: add `sink: minio` writing via `pyarrow.fs.S3FileSystem` or `minio` client; schema already long/tidy. Entry point: `src/utils/async_storage.py` + `memory_config.sink`.
2. **Remote brokers** (design: `remote_brokers_plan.md`): place federation brokers next to their federates (cuts cross-machine per-tick traffic for co-located federations). Needs remote port allocation (run port-picker via `RemoteExecutor.run`) + broker log streaming over ssh.
3. **asyncssh backend**: if machine count grows large (>10), swap RemoteExecutor internals for asyncssh persistent connections; interface stays.
4. **Docker provisioning**: replace conda-env assumption with image pull for reproducibility.
5. **RL federate remote**: lift v1 `ValueError` — needs GPU-aware placement + checkpoint path collection.
