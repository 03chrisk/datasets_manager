from preprocessingABC import PreprocessingTechniqueABC
import numpy as np
import matplotlib.pyplot as plt
from joinedDataset import JoinedDataset
from PIL import Image
from treeDataset import TreeDataset


class RandomCrop(PreprocessingTechniqueABC):
    def __init__(self, width: int, height: int):
        self.height = height
        self.width = width

    def __call__(self, image: Image) -> Image:
        w, h = image.size
        top = np.random.randint(0, h - self.height + 1)
        left = np.random.randint(0, w - self.width + 1)
        bottom = top + self.height
        right = left + self.width
        return image.crop((left, top, right, bottom))


if __name__ == "__main__":
    pathh = r"datasets\image\classification"
    dataset = TreeDataset(root=pathh, data_type='image',
                            loading_method="eager")
    print(dataset.data[0])

    plt.imshow(dataset.data[0])
    plt.axis("off")  # Turn off axis numbers
    plt.show()

    randomcrop = RandomCrop(20, 20)
    cropped = randomcrop(dataset.data[0])
    print(cropped)
    plt.imshow(cropped)
    plt.axis("off")  # Turn off axis numbers
    plt.show()