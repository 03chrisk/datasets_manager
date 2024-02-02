import os
import glob
from PIL import Image
from dataset import Dataset
import matplotlib.pyplot as plt


class TreeDataset(Dataset):
    def __init__(self, root, data_type='image',
                 loading_method=None, data=None, labels=None):
        super().__init__(root, data_type, loading_method, data, labels)

    def _load_data(self):
        extension, load_method = self._get_extension_and_loader()
        for class_dir in os.listdir(self.root):
            class_path = os.path.join(self.root, class_dir)
            if os.path.isdir(class_path):
                for filepath in glob.glob(os.path.join(class_path,
                                                        extension)):
                    self._handle_load_method(load_method, filepath)
                    self.labels.append(class_dir)


if __name__ == "__main__":
    path = r'datasets\audio\classification'

    dataset = TreeDataset(root=path, data_type='audio', loading_method='eager')
    print(len(dataset))
    image, label = dataset[0]
    print(image)
    print(label)
    # plt.imshow(image)
    # plt.axis('off')  # Turn off axis numbers
    # plt.show()
