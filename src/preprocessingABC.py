from abc import ABC, abstractmethod


class PreprocessingTechniqueABC(ABC):
    @abstractmethod
    def __call__(self, data):
        """
        Abstract method to apply the preprocessing technique to data.

        Args:
            data: The input data to be preprocessed. Either an Image.image file
            or a Tuple[np.ndarray, int]

        Returns:
            The preprocessed data. Either an Image.image file
            or a Tuple[np.ndarray, int]
        """
        pass
