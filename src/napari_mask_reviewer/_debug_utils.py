import functools
import logging
import traceback
import psutil
import gc
import sys
from typing import Callable, Any
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)


def setup_debug_logging():
    """Configure debug logging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='napari_plugin_debug.log'
    )

    # Also log to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def track_memory(location: str = "unknown"):
    """Track memory usage at a specific location"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    logger.debug(f"Memory usage at {location}: {memory_mb:.2f} MB")
    return memory_mb


@contextmanager
def memory_tracker(location: str):
    """Context manager to track memory changes"""
    start_mem = track_memory(f"{location} - start")
    yield
    end_mem = track_memory(f"{location} - end")
    diff = end_mem - start_mem
    logger.debug(f"Memory change at {location}: {diff:.2f} MB")


def debug_method(func: Callable) -> Callable:
    """Decorator to add debugging to methods"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            location = f"{func.__module__}.{func.__qualname__}"
            logger.debug(f"Entering {location}")
            with memory_tracker(location):
                result = func(*args, **kwargs)
            logger.debug(f"Exiting {location}")
            return result
        except Exception as e:
            logger.error(f"Error in {location}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    return wrapper


def monitor_gc():
    """Monitor garbage collection"""
    gc.set_debug(gc.DEBUG_LEAK)

    def print_gc_stats():
        count = gc.get_count()
        logger.debug(f"GC count: {count}")
        logger.debug(f"Garbage objects: {len(gc.garbage)}")

        for obj in gc.garbage:
            logger.debug(f"Garbage object: {type(obj)}")

    return print_gc_stats


def check_qt_objects(widget):
    """Check Qt widget hierarchy for potential issues"""
    from qtpy.QtCore import QObject

    def check_widget(obj, depth=0):
        logger.debug(f"{'  ' * depth}Checking {type(obj).__name__}")
        if hasattr(obj, 'children'):
            for child in obj.children():
                check_widget(child, depth + 1)

    check_widget(widget)


class MemoryLeakDetector:
    """Detect potential memory leaks"""

    def __init__(self):
        self.snapshots = {}

    def take_snapshot(self, name: str):
        """Take a snapshot of current objects"""
        self.snapshots[name] = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            self.snapshots[name][obj_type] = self.snapshots[name].get(obj_type, 0) + 1

    def compare_snapshots(self, name1: str, name2: str):
        """Compare two snapshots"""
        if name1 not in self.snapshots or name2 not in self.snapshots:
            logger.error("Snapshot not found")
            return

        snapshot1 = self.snapshots[name1]
        snapshot2 = self.snapshots[name2]

        for obj_type in set(snapshot1.keys()) | set(snapshot2.keys()):
            count1 = snapshot1.get(obj_type, 0)
            count2 = snapshot2.get(obj_type, 0)
            if count1 != count2:
                logger.debug(f"Object count changed for {obj_type}: {count1} -> {count2}")


# Example usage in your plugin:
"""
from debug_utils import setup_debug_logging, debug_method, MemoryLeakDetector

class MaskReviewer(QWidget):
    def __init__(self, napari_viewer):
        setup_debug_logging()
        self.leak_detector = MemoryLeakDetector()
        super().__init__()

    @debug_method
    def load_file_pair(self, index):
        self.leak_detector.take_snapshot('before_load')
        # Your existing code
        self.leak_detector.take_snapshot('after_load')
        self.leak_detector.compare_snapshots('before_load', 'after_load')
"""