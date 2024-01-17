from PIL import Image
import os
from dataset import Dataset
import glob
import matplotlib.pyplot as plt
import csv


class JoinedDataset(Dataset):
    def __init__(self, root, data_type, loading_method=None,
                 load_labels=False, data=None, labels=None):

        self.label_path = os.path.join(os.path.dirname(root), "labels.csv")
        self.load_labels = load_labels
        super().__init__(root, data_type, loading_method, data, labels)

        if self.load_labels and labels is None:
            self.load_labels_from_csv()

    def load_data(self):
        extension, load_method = self.get_extension_and_loader()

        for filepath in glob.glob(os.path.join(self.root, extension)):
            self.handle_load_method(load_method, filepath)

    def load_labels_from_csv(self):
        """
        Load labels from a CSV file.
        """
        with open(self.label_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            self.labels = [row[1] for row in reader]


if __name__ == "__main__":
    path = r"datasets\audio\regression\audio"
    image_dataset = JoinedDataset(root=path, data_type='audio',
                                  loading_method="eager", load_labels=True)
    image, label = image_dataset[0]  # Get the first image

    print(image, label)
    print(len(image_dataset))

    # Display the image
    #plt.imshow(image)
    #plt.axis("off")  # Turn off axis numbers
    #plt.show()

    train, test = image_dataset.split(train_size=0.5)
    print(len(train), len(test))

    lazy_image_dataset = JoinedDataset(
        root=path, data_type='audio', loading_method="lazy", load_labels=False
    )
    print(len(lazy_image_dataset))
    image2 = lazy_image_dataset[0]
    print(image2)
    #plt.imshow(image2)
    #plt.axis("off")  # Turn off axis numbers
    #plt.show()
    # Access and load an image lazily
