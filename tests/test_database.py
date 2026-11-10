"""Unit tests for database models, engine, and session."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database.models import Base, ModelRecord, DatasetRecord, EvaluationRecord, TrainingJobRecord


def test_database_tables_creation():
    test_engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(test_engine)

    Session = sessionmaker(bind=test_engine)
    session = Session()

    # Add a model record
    model = ModelRecord(
        id="test-model",
        name="Test Model 1B",
        provider="huggingface",
        parameter_count_b=1.0,
        context_window=2048,
        precision="fp16",
    )
    session.add(model)
    session.commit()

    retrieved = session.query(ModelRecord).filter_by(id="test-model").first()
    assert retrieved is not None
    assert retrieved.name == "Test Model 1B"
    assert retrieved.parameter_count_b == 1.0

    session.close()

