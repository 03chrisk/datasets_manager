from abc import ABC, abstractmethod
import random
import librosa
from PIL import Image
import numpy as np


class Dataset(ABC):
    def __init__(self, root, data_type='image', loading_method=None, data=None, labels=None):
        self._root = root
        self._data_type = data_type
        self._loading_method = loading_method
        self._data = data if data is not None else []
        self._labels = labels if labels is not None else []

        if data is None:
            self.load_data()

    @property
    def root(self):
        return self._root

    @property
    def data_type(self):
        return self._data_type

    @property
    def loading_method(self):
        return self._loading_method

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value

    @abstractmethod
    def load_data(self):
        pass

    def handle_load_method(self, load_method, filepath):
        if self.loading_method == "eager":
            data = load_method(filepath)
            self.data.append(data)
        elif self.loading_method == "lazy":
            self.data.append(filepath)

    def __getitem__(self, index):
        """
        Return the image at the specified index.
        For lazy loading, the image is loaded when accessed.
        """
        _, loading_method = self.get_extension_and_loader()
        data = self.data[index]
        label = self.labels[index] if self.labels else None
        if self.loading_method == "eager":
            return (data, label) if label is not None else data
        else:
            data = loading_method(data)
            return (data, label) if label is not None else data

    def load_image(self, filepath):
        try:
            return Image.open(filepath).convert("RGB")
        except IOError as e:
            print(f"Error loading image {filepath}: {e}")
            return None

    def load_audio(self, filepath):
        try:
            audio_ts, sr = librosa.load(filepath, sr=None)
            return audio_ts, sr
        except Exception as e:
            print(f"Error loading audio {filepath}: {e}")
            return None, None

    def get_extension_and_loader(self):
        """
        Determine the file extension and loading method based on the data type.
        Returns a tuple of (file extension, load method).
        """
        if self.data_type == 'image':
            return ('*.jpg', self.load_image)
        elif self.data_type == 'audio':
            return ('*.wav', self.load_audio)
        else:
            raise ValueError("Invalid data type specified")

    def __len__(self):
        return len(self.data)

    def split(self, train_size=0.8):
        """
        Split the dataset into training and test sets.
        :param train_size: The proportion of the dataset to include in the train split.
        :return: A tuple of (train_dataset, test_dataset)
        """
        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        data_shuffled, labels_shuffled = zip(*combined)

        # Calculate the split index
        split_index = int(len(data_shuffled) * train_size)

        # Split the data
        data_train = data_shuffled[:split_index]
        data_test = data_shuffled[split_index:]

        # Split the labels if they exist
        if self.labels is not None:
            labels_train = labels_shuffled[:split_index]
            labels_test = labels_shuffled[split_index:]
        else:
            labels_train, labels_test = None, None

        # Create new instances of ImageDataset for train and test sets
        train_dataset = self.__class__(
            self.root,
            data_type=self.data_type,
            data=data_train,
            labels=labels_train,
            loading_method=self.loading_method,
        )
        test_dataset = self.__class__(
            self.root,
            data_type=self.data_type,
            data=data_test,
            labels=labels_test,
            loading_method=self.loading_method,
        )

        return train_dataset, test_dataset
