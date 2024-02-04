from preprocessingABC import PreprocessingTechniqueABC
from typing import Callable, Tuple
from PIL import Image
import numpy as np


class PreprocessingPipeline(PreprocessingTechniqueABC):
    def __init__(self, *steps: Callable) -> None:
        if not all(isinstance(step,
                              PreprocessingTechniqueABC) for step in steps):
            raise ValueError("All steps must be callable")
        self._steps = steps

    @property
    def steps(self):
        return self._steps

    def __call__(self,
                 data: Image.Image | Tuple[np.ndarray, int]
                 ) -> Image.Image | Tuple[np.ndarray, int]:
        """
        Applies the preprocessing pipeline to the input data.

        Args:
            data: The input data to be preprocessed.

        Returns:
            Preprocessed data
        """
        for step in self.steps:
            data = step(data)
        return data
