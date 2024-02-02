from abc import ABC, abstractmethod


class PreprocessingTechniqueABC(ABC):
    @abstractmethod
    def __call__(self, data):
        pass
