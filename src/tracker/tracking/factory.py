"""Backend selection for the tracking layer."""
from tracker.tracking.base import TrackerBackend
from tracker.config.settings import TRACKER_BACKEND

def make_tracker(backend: str | None = None, **kwargs) -> TrackerBackend:
    """Construct a TrackerBackend by name ("sam2" | "sam3"). Does not load weights."""
    name = (backend or TRACKER_BACKEND).lower()
    if name == "sam2":
        from tracker.tracking.sam2 import Sam2Tracker
        return Sam2Tracker(**kwargs)
    if name == "sam3":
        from tracker.tracking.sam3 import Sam3Tracker
        return Sam3Tracker(**kwargs)
    raise ValueError(f"unknown tracker backend: {name!r} (expected 'sam2' or 'sam3')")
