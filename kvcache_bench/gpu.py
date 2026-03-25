"""GPU detection and VRAM monitoring."""

import subprocess
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class GpuInfo:
    name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    driver_version: str


def detect_gpu() -> Optional[GpuInfo]:
    """Detect NVIDIA GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return None
        parts = result.stdout.strip().split(",")
        if len(parts) < 5:
            return None
        return GpuInfo(
            name=parts[0].strip(),
            vram_total_mb=int(float(parts[1].strip())),
            vram_used_mb=int(float(parts[2].strip())),
            vram_free_mb=int(float(parts[3].strip())),
            driver_version=parts[4].strip(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def measure_vram() -> int:
    """Get current VRAM usage in MB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(float(result.stdout.strip()))
    except Exception:
        return 0


class VramTracker:
    """Track VRAM usage over a time period."""

    def __init__(self):
        self.baseline = 0
        self.peak = 0
        self._samples = []

    def start(self):
        self.baseline = measure_vram()
        self.peak = self.baseline
        self._samples = [self.baseline]

    def sample(self):
        v = measure_vram()
        self._samples.append(v)
        self.peak = max(self.peak, v)

    def stop(self) -> dict:
        self.sample()
        return {
            "baseline_mb": self.baseline,
            "peak_mb": self.peak,
            "delta_mb": self.peak - self.baseline,
            "samples": len(self._samples),
        }
