"""Migrate OpenLLM BenchLab data from SQLite to PostgreSQL (or between any two SQLAlchemy DBs).

Usage:
    python scripts/migrate_sqlite_to_postgres.py \
        --source sqlite:///./benchlab.db \
        --target postgresql://benchlab_user:benchlab_password@localhost:5432/benchlab_db
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Type

from sqlalchemy.orm import sessionmaker

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.database.models import (
    ApiKeyRecord,
    AsyncJobRecord,
    Base,
    BenchmarkRecord,
    DatasetRecord,
    EvaluationRecord,
    ExperimentRecord,
    ModelRecord,
    QuantizationRecord,
    TrainingJobRecord,
    UserRecord,
    WorkspaceMemberRecord,
    WorkspaceRecord,
)
from backend.app.database.session import (
    build_engine,
    ensure_default_workspace,
    init_db,
    normalize_database_url,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchlab.migrate")

ORDERED_MODELS: List[Type] = [
    UserRecord,
    WorkspaceRecord,
    WorkspaceMemberRecord,
    ApiKeyRecord,
    ModelRecord,
    DatasetRecord,
    ExperimentRecord,
    EvaluationRecord,
    TrainingJobRecord,
    QuantizationRecord,
    BenchmarkRecord,
    AsyncJobRecord,
]


def migrate_database(
    source_url: str,
    target_url: str,
    truncate_target: bool = False,
) -> Dict[str, Dict[str, int]]:
    """Copy all records from source DB to target DB in dependency order."""
    src_engine = build_engine(normalize_database_url(source_url))
    dst_engine = build_engine(normalize_database_url(target_url))

    # Ensure schema exists on target
    init_db(dst_engine)

    SrcSession = sessionmaker(bind=src_engine)
    DstSession = sessionmaker(bind=dst_engine)

    src = SrcSession()
    dst = DstSession()
    report: Dict[str, Dict[str, int]] = {}

    try:
        if truncate_target:
            for model_cls in reversed(ORDERED_MODELS):
                dst.query(model_cls).delete()
            dst.commit()

        for model_cls in ORDERED_MODELS:
            table_name = model_cls.__tablename__
            rows = src.query(model_cls).all()
            inserted = 0
            skipped = 0

            existing_ids = {
                r[0] for r in dst.query(model_cls.id).all()
            }

            columns = [col.name for col in model_cls.__table__.columns]
            for row in rows:
                row_id = getattr(row, "id")
                if row_id in existing_ids:
                    skipped += 1
                    continue
                payload = {col: getattr(row, col) for col in columns}
                dst.add(model_cls(**payload))
                existing_ids.add(row_id)
                inserted += 1

            dst.commit()
            report[table_name] = {
                "source_count": len(rows),
                "inserted": inserted,
                "skipped": skipped,
            }
            logger.info(
                f"Table '{table_name}': source={len(rows)}, inserted={inserted}, skipped={skipped}"
            )

        ensure_default_workspace(dst)
        return report
    except Exception:
        dst.rollback()
        raise
    finally:
        src.close()
        dst.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate BenchLab data from SQLite to PostgreSQL.")
    parser.add_argument(
        "--source",
        default="sqlite:///./benchlab.db",
        help="Source SQLAlchemy URL (default: sqlite:///./benchlab.db)",
    )
    parser.add_argument(
        "--target",
        default=os.getenv("DATABASE_URL", ""),
        help="Target PostgreSQL SQLAlchemy URL (or set DATABASE_URL env var)",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Clear target tables before migrating.",
    )
    args = parser.parse_args()

    if not args.target:
        parser.error("--target or DATABASE_URL environment variable is required.")

    migrate_database(args.source, args.target, truncate_target=args.truncate)


if __name__ == "__main__":
    main()

