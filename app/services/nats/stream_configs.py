from app.services.nats.nats_client import NatsStreamConfig
import os

NATS_STREAM_CONFIGS = [
    NatsStreamConfig(
        name=os.getenv("INGEST_STREAM"),
        subjects=[os.getenv("INGEST_PGVECTOR_SUBJECT"), os.getenv("INGEST_KG_SUBJECT")],
        storage="file",
        max_msgs=1000,
        max_bytes=1024 * 1024 * 10,  # 10MB
        max_age=3600,  # 1 hour
        retention="limits",
        discard="old",
        duplicate_window=0
    )
]