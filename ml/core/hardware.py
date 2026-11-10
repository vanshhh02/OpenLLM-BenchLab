"""Hardware detection and resource monitoring utility for OpenLLM BenchLab.

Provides unified detection across Apple Silicon (Metal/MPS), NVIDIA CUDA GPUs,
and multi-core CPU architectures with memory tracking and capability inspection.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger("benchlab.hardware")


@dataclass
class HardwareProfile:
    platform_system: str
    platform_release: str
    platform_machine: str
    cpu_brand: str
    cpu_count_physical: int
    cpu_count_logical: int
    total_ram_gb: float
    available_ram_gb: float
    accelerator_type: str  # 'cuda', 'mps', or 'cpu'
    accelerator_name: str
    accelerator_count: int
    accelerator_memory_gb: Optional[float] = None
    bf16_supported: bool = False
    fp16_supported: bool = True
    int8_supported: bool = True
    int4_bitsandbytes_supported: bool = False
    vllm_supported: bool = False
    llamacpp_supported: bool = True
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_cpu_brand() -> str:
    """Retrieve human-readable CPU brand string."""
    system = platform.system()
    try:
        if system == "Darwin":
            res = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            )
            brand = res.stdout.strip()
            if not brand:
                res2 = subprocess.run(
                    ["sysctl", "-n", "hw.model"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                brand = res2.stdout.strip()
            return brand or "Apple Silicon"
        elif system == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
    except Exception as e:
        logger.debug(f"Could not read CPU brand: {e}")
    return platform.processor() or "Unknown CPU"


def detect_hardware() -> HardwareProfile:
    """Inspect the current runtime environment and return the HardwareProfile."""
    system = platform.system()
    machine = platform.machine()
    mem = psutil.virtual_memory()
    total_ram_gb = round(mem.total / (1024**3), 2)
    avail_ram_gb = round(mem.available / (1024**3), 2)

    cpu_brand = get_cpu_brand()
    phys_cores = psutil.cpu_count(logical=False) or 1
    log_cores = psutil.cpu_count(logical=True) or 1

    accelerator_type = "cpu"
    accelerator_name = "CPU"
    accelerator_count = 0
    accelerator_memory_gb = None
    bf16_supported = False
    fp16_supported = True
    int8_supported = True
    int4_bnb_supported = False
    vllm_supported = False
    llamacpp_supported = True
    notes: List[str] = []

    # Check PyTorch device availability if torch is importable
    try:
        import torch

        if torch.cuda.is_available():
            accelerator_type = "cuda"
            accelerator_count = torch.cuda.device_count()
            accelerator_name = torch.cuda.get_device_name(0)
            accelerator_memory_gb = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
            bf16_supported = torch.cuda.is_bf16_supported()
            fp16_supported = True
            vllm_supported = system == "Linux"

            # Check bitsandbytes CUDA support
            try:
                import bitsandbytes as bnb
                int4_bnb_supported = True
            except Exception:
                int4_bnb_supported = False
                notes.append("bitsandbytes CUDA extension not found or failed to load.")

            notes.append(f"CUDA accelerator detected: {accelerator_name} ({accelerator_memory_gb} GB VRAM).")

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator_type = "mps"
            accelerator_count = 1
            accelerator_name = f"{cpu_brand} (Apple Metal MPS)"
            accelerator_memory_gb = total_ram_gb  # Unified memory
            bf16_supported = hasattr(torch.backends.mps, "is_built") and torch.backends.mps.is_built()
            fp16_supported = True
            vllm_supported = False  # vLLM is strictly Linux CUDA/ROCm
            llamacpp_supported = True
            int4_bnb_supported = False  # bitsandbytes 4-bit NF4 kernels target CUDA GPUs
            notes.append(
                "Apple Silicon unified memory detected. Acceleration via Metal Performance Shaders (MPS). "
                "PyTorch dynamic quantization, FP16/BF16, and llama.cpp Metal GGUF are supported."
            )
        else:
            accelerator_type = "cpu"
            accelerator_name = f"{cpu_brand} (CPU Fallback)"
            notes.append("No hardware accelerator found; running in CPU fallback mode.")
    except ImportError:
        # PyTorch not yet imported or installed
        if system == "Darwin" and machine in ("arm64", "aarch64"):
            accelerator_type = "mps"
            accelerator_name = f"{cpu_brand} (Apple Silicon)"
            accelerator_memory_gb = total_ram_gb
            notes.append("Apple Silicon detected (torch pending initialization).")
        else:
            accelerator_type = "cpu"
            accelerator_name = cpu_brand

    return HardwareProfile(
        platform_system=system,
        platform_release=platform.release(),
        platform_machine=machine,
        cpu_brand=cpu_brand,
        cpu_count_physical=phys_cores,
        cpu_count_logical=log_cores,
        total_ram_gb=total_ram_gb,
        available_ram_gb=avail_ram_gb,
        accelerator_type=accelerator_type,
        accelerator_name=accelerator_name,
        accelerator_count=accelerator_count,
        accelerator_memory_gb=accelerator_memory_gb,
        bf16_supported=bf16_supported,
        fp16_supported=fp16_supported,
        int8_supported=int8_supported,
        int4_bitsandbytes_supported=int4_bnb_supported,
        vllm_supported=vllm_supported,
        llamacpp_supported=llamacpp_supported,
        notes=notes,
    )


class MemoryTracker:
    """Context manager and utility to track RAM and GPU/MPS memory consumption."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.start_ram_bytes: int = 0
        self.end_ram_bytes: int = 0
        self.peak_ram_bytes: int = 0
        self.start_vram_bytes: Optional[int] = None
        self.peak_vram_bytes: Optional[int] = None
        self.end_vram_bytes: Optional[int] = None

    def __enter__(self) -> "MemoryTracker":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        process = psutil.Process(os.getpid())
        self.start_ram_bytes = process.memory_info().rss
        self.peak_ram_bytes = self.start_ram_bytes

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                self.start_vram_bytes = torch.cuda.memory_allocated()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
                self.start_vram_bytes = torch.mps.current_allocated_memory()
        except Exception:
            pass

    def update_peak(self):
        process = psutil.Process(os.getpid())
        current_ram = process.memory_info().rss
        if current_ram > self.peak_ram_bytes:
            self.peak_ram_bytes = current_ram

        try:
            import torch
            if torch.cuda.is_available():
                current_vram = torch.cuda.max_memory_allocated()
                if self.peak_vram_bytes is None or current_vram > self.peak_vram_bytes:
                    self.peak_vram_bytes = current_vram
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
                current_vram = torch.mps.current_allocated_memory()
                if self.peak_vram_bytes is None or current_vram > self.peak_vram_bytes:
                    self.peak_vram_bytes = current_vram
        except Exception:
            pass

    def stop(self):
        self.update_peak()
        process = psutil.Process(os.getpid())
        self.end_ram_bytes = process.memory_info().rss

        try:
            import torch
            if torch.cuda.is_available():
                self.end_vram_bytes = torch.cuda.memory_allocated()
                self.peak_vram_bytes = torch.cuda.max_memory_allocated()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
                self.end_vram_bytes = torch.mps.current_allocated_memory()
        except Exception:
            pass

    def get_summary(self) -> Dict[str, Any]:
        delta_ram_mb = round((self.end_ram_bytes - self.start_ram_bytes) / (1024**2), 2)
        peak_ram_mb = round(self.peak_ram_bytes / (1024**2), 2)
        summary: Dict[str, Any] = {
            "ram_delta_mb": delta_ram_mb,
            "peak_ram_mb": peak_ram_mb,
        }
        if self.peak_vram_bytes is not None:
            summary["peak_vram_mb"] = round(self.peak_vram_bytes / (1024**2), 2)
        if self.start_vram_bytes is not None and self.end_vram_bytes is not None:
            summary["vram_delta_mb"] = round((self.end_vram_bytes - self.start_vram_bytes) / (1024**2), 2)
        return summary

