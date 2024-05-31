import abc

import numpy as np


class BaseDetector(abc.ABC):
    """Base Face Detector"""

    @abc.abstractmethod
    def _execute(
        self,
        image_array: np.ndarray,
        expected_count: int | None = None,
    ) -> tuple[np.ndarray, ...]:
        raise NotImplementedError("_execute")

    def __call__(self, *args, **kwargs) -> tuple[np.ndarray, ...]:
        return self._execute(*args, **kwargs)
