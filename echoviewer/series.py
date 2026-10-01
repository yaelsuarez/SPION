"""Cached access to one sample's volumes across echo times.

Each volume in this dataset is ~22 MB and takes ~0.6 s to stack from its 128
DICOM slices (~0.1 s once the OS file cache is warm). Twenty-six echo times
would be ~560 MB if held all at once, so :class:`EchoSeries` keeps a bounded
LRU cache and warms the neighbours of the current echo in a background thread.

The result is that dragging the echo slider almost always hits a cached
volume; only a jump to a far-away echo pays the load cost.

Threading note: the worker only ever *fills* the cache. It never touches Qt
and never hands objects to the UI directly - the UI thread just calls
:meth:`EchoSeries.volume` again and finds the entry already there. Volumes are
treated as immutable once built, so sharing them between threads is safe.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

from .dataset import EchoTime, get_available_echotimes, load_volume_from_folder

from dicomview.volume import Volume

#: How many volumes to keep in memory. Eight is ~175 MB for this dataset.
DEFAULT_CACHE_SIZE = 8

#: How many echo times either side of the current one to warm in advance.
PREFETCH_RADIUS = 2


class EchoSeries:
    """One sample's volumes, indexed by echo time, with caching and prefetch.

    Args:
        name: Sample name, e.g. ``"Syringes"``.
        echotimes: The sample's echo times, in ascending order.
        cache_size: Maximum number of volumes held in memory at once.

    Raises:
        ValueError: If ``echotimes`` is empty.
    """

    def __init__(self, name: str, echotimes: list[EchoTime], cache_size: int = DEFAULT_CACHE_SIZE):
        if not echotimes:
            raise ValueError(f"Sample {name!r} has no echo times")
        self.name = name
        self.echotimes = list(echotimes)
        self._cache_size = max(1, cache_size)
        self._cache: OrderedDict[int, Volume] = OrderedDict()
        self._lock = threading.Lock()

        # Background prefetch state.
        self._wanted: list[int] = []
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

    # -- construction ------------------------------------------------------

    @classmethod
    def from_sample(
        cls,
        volumes_root: str | Path,
        sample: str,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> "EchoSeries":
        """Build a series by scanning ``volumes_root/sample`` for echo times."""
        sample_path = Path(volumes_root) / sample
        return cls(sample, get_available_echotimes(sample_path), cache_size=cache_size)

    # -- basic access ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.echotimes)

    def echotime(self, index: int) -> EchoTime:
        """The :class:`~echoviewer.dataset.EchoTime` at ``index``."""
        return self.echotimes[index]

    def index_of_nearest(self, echo_time: float) -> int:
        """Index of the echo time closest to ``echo_time`` milliseconds."""
        return min(
            range(len(self.echotimes)),
            key=lambda i: abs(self.echotimes[i].value - echo_time),
        )

    def is_cached(self, index: int) -> bool:
        """True if ``index`` can be served without touching disk."""
        with self._lock:
            return index in self._cache

    @property
    def cached_indices(self) -> list[int]:
        """Currently cached echo indices, oldest first."""
        with self._lock:
            return list(self._cache)

    def volume(self, index: int) -> Volume:
        """Return the volume for echo ``index``, loading it if necessary.

        Args:
            index: Position in :attr:`echotimes`.

        Returns:
            The stacked volume.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if not 0 <= index < len(self.echotimes):
            raise IndexError(f"echo index {index} out of range [0, {len(self.echotimes)})")

        with self._lock:
            cached = self._cache.get(index)
            if cached is not None:
                self._cache.move_to_end(index)
                return cached

        # Load outside the lock so a slow read never blocks the UI thread's
        # cache lookups (or the prefetch worker's).
        volume = load_volume_from_folder(self.echotimes[index].path)
        self._store(index, volume)
        return volume

    def _store(self, index: int, volume: Volume) -> None:
        """Insert a volume into the cache, evicting the least recently used."""
        with self._lock:
            self._cache[index] = volume
            self._cache.move_to_end(index)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Drop every cached volume."""
        with self._lock:
            self._cache.clear()

    # -- integrity ---------------------------------------------------------

    def slice_counts(self) -> dict[int, int]:
        """Number of DICOM files in each echo folder, keyed by echo index.

        Only file headers are touched, so this is cheap enough to run when a
        sample is opened.
        """
        from dicomview.loader import find_dicom_files

        return {
            index: len(find_dicom_files(echo.path, recursive=False))
            for index, echo in enumerate(self.echotimes)
        }

    def inconsistent_echoes(self) -> list[tuple[int, int]]:
        """Echo times whose slice count differs from the rest of the sample.

        An echo folder with fewer slices than its neighbours is an incomplete
        export, not a different acquisition, and it silently changes the
        geometry when you step onto it. Returns ``(echo_index, n_slices)``
        pairs for the odd ones out, empty when the sample is uniform.
        """
        counts = self.slice_counts()
        if not counts:
            return []
        values = list(counts.values())
        typical = max(set(values), key=values.count)
        return [(index, n) for index, n in counts.items() if n != typical]

    # -- background prefetch ----------------------------------------------

    def prefetch_around(self, index: int, radius: int = PREFETCH_RADIUS) -> None:
        """Warm the cache for echo times near ``index``, in the background.

        Neighbours are queued nearest-first, so the echo the user is most
        likely to reach next is loaded first. Calling this again replaces the
        pending queue rather than adding to it, so a fast slider drag does not
        build up a backlog of stale requests.
        """
        wanted: list[int] = []
        for offset in range(1, radius + 1):
            for candidate in (index + offset, index - offset):
                if 0 <= candidate < len(self.echotimes) and not self.is_cached(candidate):
                    wanted.append(candidate)

        with self._lock:
            self._wanted = wanted
        if wanted:
            self._ensure_worker()
            self._wake.set()

    def _ensure_worker(self) -> None:
        """Start the prefetch thread if it is not already running."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(
            target=self._prefetch_loop, name=f"prefetch-{self.name}", daemon=True
        )
        self._worker.start()

    def _prefetch_loop(self) -> None:
        """Load queued echo times until asked to stop."""
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            while not self._stop.is_set():
                with self._lock:
                    pending = [i for i in self._wanted if i not in self._cache]
                    self._wanted = pending[1:] if pending else []
                if not pending:
                    break
                try:
                    index = pending[0]
                    self._store(index, load_volume_from_folder(self.echotimes[index].path))
                except Exception:
                    # A prefetch is only ever an optimisation; if it fails the
                    # UI will load the volume itself and surface the error then.
                    continue

    def close(self) -> None:
        """Stop the prefetch thread and release cached volumes."""
        self._stop.set()
        self._wake.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)
        self._worker = None
        self.clear()

    def __enter__(self) -> "EchoSeries":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
