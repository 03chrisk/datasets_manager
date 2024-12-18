from abc import ABC, abstractmethod
import random
import librosa
from PIL import Image
from typing import Optional, List, Any, Tuple, Callable
import numpy as np


class Dataset(ABC):
    def __init__(self, root: str,
                 data_type: str = "image",
                 loading_method: str = "lazy",
                 data: Optional[List[str | Image.Image | Tuple[np.ndarray,
                                                               int]]] = None,
                 labels: Optional[List[str | int]] = None) -> None:

        if not isinstance(root, str):
            raise ValueError("root must be a string")
        self._root = root

        if data_type not in ["image", "audio"]:
            raise ValueError("data_type must be in 'image' or 'audio'")
        self._data_type = data_type

        if loading_method not in ["lazy", "eager"]:
            raise ValueError("loading_method must be 'lazy' or 'eager'")
        self._loading_method = loading_method

        self._data = data if data is not None else []
        self._labels = labels if labels is not None else []

        if data is None:
            self._load_data()

    @property
    def root(self) -> str:
        return self._root

    @property
    def data_type(self) -> str:
        return self._data_type

    @property
    def loading_method(self) -> str:
        return self._loading_method

    @property
    def data(self) -> List:
        return self._data.copy()

    @data.setter
    def data(self, value) -> None:
        self._data = value

    @property
    def labels(self) -> List[str | int]:
        return self._labels

    @labels.setter
    def labels(self, value) -> None:
        self._labels = value

    @abstractmethod
    def _load_data(self) -> None:
        """
        Abstract method to load data.

        Returns:
            None
        """
        pass

    def _handle_load_method(self, load_method: Callable[[str], Any],
                            filepath: str) -> None:
        """
        Handles the loading of the data based on the loading method.

        Args:
            load_method (function): The method used to load data.

            filepath (str): The file path to the data.

        Returns:
            None
        """
        if not callable(load_method):
            raise TypeError("load_method must be callable")

        if not isinstance(filepath, str):
            raise TypeError("filepath must be a string")

        if self.loading_method == "eager":
            data = load_method(filepath)
            self._data.append(data)
        elif self.loading_method == "lazy":
            self._data.append(filepath)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Retrieves the data sample at the specified index.

        Args:
            index (int): The index of the data sample.

        Returns:
            tuple or object: The data sample and its corresponding
            label (if available).
        """
        _, loading_method = self._get_extension_and_loader()
        data = self.data[index]
        label = self.labels[index] if self.labels else None
        if self.loading_method == "eager":
            return (data, label) if label is not None else data
        else:
            data = loading_method(data)
            return (data, label) if label is not None else data

    def _load_image(self, filepath: str) -> Image:
        """
        Loads an image from the specified file path.

        Args:
            filepath (str): The file path to the image.

        Returns:
            PIL.Image.Image: The loaded image. If the loading fails,
            returns None.
        """
        try:
            return Image.open(filepath).convert("RGB")
        except IOError as e:
            print(f"Error loading image {filepath}: {e}")
            return None

    def _load_audio(self, filepath: str):
        """
        Loads audio data from the specified file path.

        Args:
            filepath (str): The file path to the audio data.

        Returns:
            tuple (np.ndarray, int): Tuple containing the audio time series
            and its sampling rate.

            If the loading fails, returns (None, None).
        """
        try:
            audio_ts, sr = librosa.load(filepath, sr=None)
            return audio_ts, sr
        except Exception as e:
            print(f"Error loading audio {filepath}: {e}")
            return None, None

    def _get_extension_and_loader(self):
        """
        Determine the file extension and whether to load an image or audio
        file based on the data type.

        Args:
            None

        Returns:
            Tuple[str, Callable] of file extension and load method
        """
        if self.data_type == "image":
            return ("*.jpg", self._load_image)
        elif self.data_type == "audio":
            return ("*.wav", self._load_audio)
        else:
            raise ValueError("Invalid data type specified")

    def __len__(self) -> int:
        """
        Args:
            None

        Returns:
            An int representing the length (number of datapoints) of the
            dataset
        """
        return len(self._data)

    def split(self, train_size: float = 0.8) -> Tuple['Dataset', 'Dataset']:
        """
        Split the dataset into training and test sets.

        Args:
            train_size: The proportion of the dataset to include
            in the train split.

        Returns:
            Tuple['Dataset', 'Dataset'] A tuple of two instances of the Dataset
            class
        """
        if not (0 < train_size < 1):
            raise ValueError("train_size must be a value between 0 and 1")

        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        data_shuffled, labels_shuffled = zip(*combined)

        split_index = int(len(data_shuffled) * train_size)

        data_train = data_shuffled[:split_index]
        data_test = data_shuffled[split_index:]

        if self.labels is not None:
            labels_train = labels_shuffled[:split_index]
            labels_test = labels_shuffled[split_index:]
        else:
            labels_train, labels_test = None, None

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
