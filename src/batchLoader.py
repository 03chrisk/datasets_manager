import numpy as np
from dataset import Dataset
from typing import Iterator


class BatchLoader:
    """
    A class for loading batches of data from a dataset.

    Attributes:
        _dataset (JoinedDataset): The dataset from which to load data.
        _batch_size (int): The size of each batch.
        _shuffle (bool): Whether to shuffle the dataset before loading batches.
        _include_last_batch (bool): Whether to include the last batch if it's
        smaller than batch_size.
    """
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True,
                 include_last_batch: bool = True) -> None:
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be an instance of Dataset or its"
                            "subclass")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be a boolean")
        if not isinstance(include_last_batch, bool):
            raise ValueError("include_last_batch must be a boolean")

        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._include_last_batch = include_last_batch

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        if not isinstance(value, int) or value <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._batch_size = value

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("shuffle must be a boolean")
        self._shuffle = value

    @property
    def include_last_batch(self) -> bool:
        return self._include_last_batch

    @include_last_batch.setter
    def include_last_batch(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("include_last_batch must be a boolean")
        self._include_last_batch = value

    def __iter__(self) -> Iterator:
        """
        Returns an iterator to iterate over batches of data.
        Args:
            None
        Returns:
            iterator: An iterator over batches of data.
        """

        dataset_size = len(self.dataset)
        indices = np.arange(dataset_size)

        if self.shuffle:
            np.random.shuffle(indices)

        num_full_batches = dataset_size // self.batch_size

        # Full batches
        for i in range(num_full_batches):
            yield [self.dataset[idx] for idx in indices[i*self.batch_size:
                                                        (i+1)*self.batch_size]]

        # Handle the last batch
        if self.include_last_batch and dataset_size % self.batch_size != 0:
            yield [self.dataset[idx] for idx in indices[num_full_batches *
                                                        self.batch_size:]]

    def __len__(self) -> int:
        """
        Calculates the number of batches.

        Args:
            None

        Returns:
            int: The number of batches that will be produced by the iterator.
        """
        dataset_size = len(self.dataset)
        if self.include_last_batch and dataset_size % self.batch_size != 0:
            return dataset_size // self.batch_size + 1
        return dataset_size // self.batch_size
