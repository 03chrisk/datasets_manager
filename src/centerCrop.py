from preprocessingABC import PreprocessingTechniqueABC
from PIL import Image


class CenterCrop(PreprocessingTechniqueABC):
    def __init__(self, width: int, height: int) -> None:
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
        Performs center cropping on the input image.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: The center-cropped image.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("The input must be a PIL.Image.Image")

        w, h = image.size

        top = max(0, (h - self.height) // 2)
        left = max(0, (w - self.width) // 2)
        bottom = min(h, top + self.height)
        right = min(w, left + self.width)

        return image.crop((left, top, right, bottom))
