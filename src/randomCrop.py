from preprocessingABC import PreprocessingTechniqueABC
import numpy as np
from PIL import Image


class RandomCrop(PreprocessingTechniqueABC):
    def __init__(self, width: int, height: int):
        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError("Width and height must be integers")
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive integers")

        self._height = height
        self._width = width

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    def __call__(self, image: Image) -> Image:
        """
        Performs random cropping on the input image.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: The randomly cropped image.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("The input must be a PIL.Image.Image")

        w, h = image.size
        top = np.random.randint(0, h - self.height + 1)
        left = np.random.randint(0, w - self.width + 1)
        bottom = top + self.height
        right = left + self.width
        return image.crop((left, top, right, bottom))
