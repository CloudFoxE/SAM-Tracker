"""Memory management utilities for video processing.

This module centralizes all memory cleanup operations to prevent memory
accumulation during batch video processing.
"""

import logging
import gc
import shutil
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:
    torch = None

try:
    import psutil
except ImportError:
    psutil = None

from tracker.config.settings import DEFAULT_MEMORY_CONFIG

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages memory cleanup operations during video processing.

    Centralizes garbage collection, GPU memory clearing, and temp file cleanup.
    """

    def __init__(self,
                 cleanup_interval: int = DEFAULT_MEMORY_CONFIG.cleanup_interval_frames,
                 gc_passes: int = DEFAULT_MEMORY_CONFIG.gc_passes,
                 enable_gpu_cleanup: bool = DEFAULT_MEMORY_CONFIG.enable_gpu_cleanup):
        """Initialize memory manager.

        Args:
            cleanup_interval: Run periodic cleanup every N frames
            gc_passes: Number of garbage collection passes per cleanup
            enable_gpu_cleanup: Whether to clear GPU cache during cleanup
        """
        self.cleanup_interval = cleanup_interval
        self.gc_passes = gc_passes
        self.enable_gpu_cleanup = enable_gpu_cleanup

    def should_cleanup(self, frame_idx: int) -> bool:
        """Check if periodic cleanup should run at this frame.

        Args:
            frame_idx: Current frame index

        Returns:
            True if cleanup should run
        """
        return frame_idx > 0 and frame_idx % self.cleanup_interval == 0

    def periodic_cleanup(self):
        """Run periodic cleanup during long video processing."""
        try:
            # Garbage collection
            gc.collect()

            # Clear GPU cache if enabled
            if self.enable_gpu_cleanup and torch and torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.debug(f"Periodic cleanup executed")

        except Exception as e:
            logger.warning(f"Periodic cleanup warning: {e}")

    def cleanup_after_video(self, temp_dir: Optional[Path] = None):
        """Full cleanup after processing a video.

        Args:
            temp_dir: Optional temporary directory to delete
        """
        try:
            # Multiple garbage collection passes
            for _ in range(self.gc_passes):
                gc.collect()

            # Clear GPU memory
            if self.enable_gpu_cleanup:
                self.clear_gpu_memory()

            # Delete temporary directory
            if temp_dir and temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                    logger.info(f"Deleted temporary directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Could not delete temp directory: {e}")

            logger.debug("Post-video cleanup complete")

        except Exception as e:
            logger.warning(f"Cleanup after video warning: {e}")

    def clear_gpu_memory(self):
        """Clear GPU memory cache."""
        try:
            if torch and torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                # Additional cleanup methods if available
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()

                logger.debug("Cleared GPU memory cache")

        except Exception as e:
            logger.warning(f"GPU memory cleanup warning: {e}")

    def get_memory_usage(self) -> tuple:
        """Get current memory usage.

        Returns:
            Tuple of (gpu_memory_mb, ram_mb)
        """
        gpu_mb = 0.0
        ram_mb = 0.0

        try:
            # GPU memory
            if torch and torch.cuda.is_available():
                gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)

            # RAM usage
            if psutil:
                try:
                    process = psutil.Process()
                    ram_mb = process.memory_info().rss / (1024 * 1024)
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Could not retrieve memory usage: {e}")

        return (gpu_mb, ram_mb)

    def log_memory_usage(self, context: str = ""):
        """Log current memory usage.

        Args:
            context: Optional context string for logging
        """
        gpu_mb, ram_mb = self.get_memory_usage()
        context_str = f" ({context})" if context else ""
        logger.info(f"Memory usage{context_str}: GPU={gpu_mb:.1f}MB, RAM={ram_mb:.1f}MB")

    def reset_gpu_stats(self):
        """Reset GPU memory statistics."""
        try:
            if torch and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                    torch.cuda.reset_accumulated_memory_stats()
                logger.debug("Reset GPU memory statistics")

        except Exception as e:
            logger.warning(f"Could not reset GPU stats: {e}")

    def full_cleanup(self, temp_dir: Optional[Path] = None, reset_stats: bool = False):
        """Comprehensive cleanup including all memory and statistics.

        Args:
            temp_dir: Optional temporary directory to delete
            reset_stats: Whether to reset GPU memory statistics
        """
        try:
            # Standard cleanup
            self.cleanup_after_video(temp_dir)

            # Reset statistics if requested
            if reset_stats:
                self.reset_gpu_stats()

            logger.debug("Full cleanup complete")

        except Exception as e:
            logger.warning(f"Full cleanup warning: {e}")
