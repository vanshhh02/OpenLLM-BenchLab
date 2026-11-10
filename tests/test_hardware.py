"""Unit tests for hardware detection and memory tracking."""

import platform
import pytest
from ml.core.hardware import detect_hardware, MemoryTracker, get_cpu_brand


def test_hardware_detection_fields():
    profile = detect_hardware()
    assert profile.platform_system == platform.system()
    assert profile.total_ram_gb > 0
    assert profile.available_ram_gb > 0
    assert profile.cpu_count_physical >= 1
    assert profile.cpu_count_logical >= 1
    assert profile.accelerator_type in ["cuda", "mps", "cpu"]
    assert isinstance(profile.notes, list)


def test_cpu_brand_non_empty():
    brand = get_cpu_brand()
    assert isinstance(brand, str)
    assert len(brand.strip()) > 0


def test_memory_tracker_context_manager():
    tracker = MemoryTracker()
    with tracker:
        # Allocate some memory
        data = [i for i in range(100000)]
        del data
    summary = tracker.get_summary()
    assert "ram_delta_mb" in summary
    assert "peak_ram_mb" in summary
    assert summary["peak_ram_mb"] > 0

