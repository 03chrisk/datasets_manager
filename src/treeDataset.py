import os
import glob
from src.dataset import Dataset
from typing import Optional, List, Any


class TreeDataset(Dataset):
    def __init__(self, root: str,
                 data_type: str,
                 loading_method: str,
                 data: Optional[List[Any]] = None,
                 labels: Optional[List[Any]] = None) -> None:
        super().__init__(root, data_type, loading_method, data, labels)

    def _load_data(self) -> None:
        """
        Loads data from the disk stored in the root folder

        Args:
            None

        Returns:
            None
        """
        extension, load_method = self._get_extension_and_loader()
        for class_dir in os.listdir(self.root):
            class_path = os.path.join(self.root, class_dir)
            if os.path.isdir(class_path):
                for filepath in glob.glob(os.path.join(class_path,
                                                       extension)):
                    self._handle_load_method(load_method, filepath)
                    self.labels.append(class_dir)
