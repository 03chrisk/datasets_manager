import os
from src.dataset import Dataset
import glob
import csv
from typing import Optional, List, Any
import re


class JoinedDataset(Dataset):
    def __init__(self, root: str,
                 data_type: str,
                 loading_method: str,
                 load_labels: bool = False,
                 data: Optional[List[Any]] = None,
                 labels: Optional[List[Any]] = None) -> None:

        self._label_path = os.path.join(os.path.dirname(root), "labels.csv")

        if not isinstance(load_labels, bool):
            raise ValueError("load labels should be True or False")
        self._load_labels = load_labels

        super().__init__(root, data_type, loading_method, data, labels)

        if self.load_labels and labels is None:
            self._load_labels_from_csv()

    @property
    def label_path(self) -> str:
        return self._label_path

    @label_path.setter
    def label_path(self, value: str) -> None:
        self._label_path = value

    @property
    def load_labels(self) -> bool:
        return self._load_labels

    @load_labels.setter
    def load_labels(self, value: bool) -> None:
        self._load_labels = value

    @staticmethod
    def numerical_sort_key(s: str) -> List[int]:
        """
        A sorting key function that extracts numbers from a filename and
        sorts according to the numerical value.
        """
        parts = re.findall(r'\d+', s)
        return [int(part) for part in parts]

    def _load_data(self) -> None:
        """
        Loads data from the disk stored in the root folder

        Args:
            None

        Returns:
            None
        """
        extension, load_method = self._get_extension_and_loader()

        filepaths = glob.glob(os.path.join(self.root, extension))
        for filepath in sorted(filepaths, key=self.numerical_sort_key):
            self._handle_load_method(load_method, filepath)

    def _load_labels_from_csv(self) -> None:
        """
        Load labels from a CSV file

        Args:
            None

        Returns:
            None
        """
        with open(self.label_path, newline="") as csvfile:
            reader = csv.reader(csvfile)
            self.labels = [row[1] for row in reader]
