"""Dataset management API endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.database.models import DatasetRecord
from backend.app.database.session import get_db
from backend.app.schemas.api_schemas import (
    DatasetLoadRequest,
    DatasetStatisticsResponse,
)
from ml.datasets.billsum import BillSumDatasetManager

logger = logging.getLogger("benchlab.api.datasets")
router = APIRouter()


@router.get("", response_model=List[Dict[str, Any]])
def list_datasets(db: Session = Depends(get_db)):
    """List all registered datasets and their high-level record counts."""
    records = db.query(DatasetRecord).all()
    if not records:
        # Return default registered datasets
        return [
            {
                "id": "billsum",
                "name": "BillSum US Congressional Summarization",
                "description": "US congressional bills and legislative reference summaries",
                "train_count": 18949,
                "val_count": 1237,
                "test_count": 3269,
                "status": "ready",
            },
            {
                "id": "pubmed_qa",
                "name": "PubMedQA (Future Extension)",
                "description": "Biomedical question answering and summarization research corpus",
                "train_count": 0,
                "val_count": 0,
                "test_count": 0,
                "status": "available",
            },
        ]
    return [
        {
            "id": r.id,
            "name": r.name,
            "description": r.description,
            "train_count": r.train_count,
            "val_count": r.val_count,
            "test_count": r.test_count,
            "avg_input_tokens": r.avg_input_tokens,
            "avg_summary_tokens": r.avg_summary_tokens,
            "created_at": r.created_at,
        }
        for r in records
    ]


@router.post("/load", response_model=DatasetStatisticsResponse)
def load_dataset_endpoint(
    req: DatasetLoadRequest,
    db: Session = Depends(get_db),
):
    """Trigger dataset loading, validation, cleaning, and statistical profiling."""
    if req.dataset_name.lower() != "billsum":
        raise HTTPException(
            status_code=400,
            detail=f"Dataset '{req.dataset_name}' not supported. BillSum is the primary benchmark dataset.",
        )

    manager = BillSumDatasetManager()
    raw = manager.load_or_download_raw(
        max_train_samples=req.max_train_samples,
        max_test_samples=req.max_test_samples,
    )
    cleaned, stats = manager.prepare_and_clean_dataset(raw)
    samples = cleaned.get("ca_test", [])[:5] or cleaned.get("train", [])[:5]
    manager.sync_to_database(stats, samples)

    rec = db.query(DatasetRecord).filter_by(id="billsum").first()
    if not rec:
        raise HTTPException(status_code=500, detail="Failed to retrieve dataset record after sync.")

    return DatasetStatisticsResponse(
        id=rec.id,
        name=rec.name,
        description=rec.description,
        train_count=rec.train_count,
        val_count=rec.val_count,
        test_count=rec.test_count,
        avg_input_tokens=rec.avg_input_tokens,
        avg_summary_tokens=rec.avg_summary_tokens,
        max_input_tokens=rec.max_input_tokens,
        max_summary_tokens=rec.max_summary_tokens,
        metadata_json=rec.metadata_json or {},
        sample_records=rec.sample_records or [],
    )


@router.get("/{dataset_id}/statistics", response_model=DatasetStatisticsResponse)
def get_dataset_statistics(
    dataset_id: str,
    db: Session = Depends(get_db),
):
    """Retrieve detailed token length distributions and split statistics for a dataset."""
    rec = db.query(DatasetRecord).filter_by(id=dataset_id).first()
    if not rec:
        # Load from disk if available
        manager = BillSumDatasetManager()
        raw = manager.load_or_download_raw(max_train_samples=100, max_test_samples=50)
        cleaned, stats = manager.prepare_and_clean_dataset(raw)
        manager.sync_to_database(stats, cleaned.get("ca_test", [])[:5])
        rec = db.query(DatasetRecord).filter_by(id=dataset_id).first()

    if not rec:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    return DatasetStatisticsResponse(
        id=rec.id,
        name=rec.name,
        description=rec.description,
        train_count=rec.train_count,
        val_count=rec.val_count,
        test_count=rec.test_count,
        avg_input_tokens=rec.avg_input_tokens,
        avg_summary_tokens=rec.avg_summary_tokens,
        max_input_tokens=rec.max_input_tokens,
        max_summary_tokens=rec.max_summary_tokens,
        metadata_json=rec.metadata_json or {},
        sample_records=rec.sample_records or [],
    )

