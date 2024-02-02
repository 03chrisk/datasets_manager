from preprocessingABC import PreprocessingTechniqueABC
import numpy as np
import matplotlib.pyplot as plt
from joinedDataset import JoinedDataset


class RandomCrop(PreprocessingTechniqueABC):
    def __init__(self, width, height):
        self.height = height
        self.width = width

    def __call__(self, image):
        w, h = image.size
        top = np.random.randint(0, h - self.height + 1)
        left = np.random.randint(0, w - self.width + 1)
        bottom = top + self.height
        right = left + self.width
        return image.crop((left, top, right, bottom))


if __name__ == "__main__":
    pathh = r"datasets\image\regression\crowds"
    dataset = JoinedDataset(root=pathh, data_type='image',
                            loading_method="eager", load_labels=False)
    print(dataset.data[0])

    plt.imshow(dataset.data[0])
    plt.axis("off")  # Turn off axis numbers
    plt.show()

    randomcrop = RandomCrop(300, 300)
    cropped = randomcrop(dataset.data[0])
    print(cropped)
    plt.imshow(cropped)
    plt.axis("off")  # Turn off axis numbers
    plt.show()
