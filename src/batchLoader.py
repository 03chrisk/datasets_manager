import numpy as np
from joinedDataset import JoinedDataset
# from treeDataset import TreeDataset

# maybe ask whether iterator needs a __next__()


class BatchLoader:
    def __init__(self, dataset, batch_size, shuffle=True,
                 include_last_batch=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.include_last_batch = include_last_batch

    def __iter__(self):
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

    def __len__(self):
        dataset_size = len(self.dataset)
        if self.include_last_batch and dataset_size % self.batch_size != 0:
            return dataset_size // self.batch_size + 1
        return dataset_size // self.batch_size


if __name__ == "__main__":
    path = r"datasets\audio\regression\audio"
    pathh = r"datasets\image\regression\crowds"
    dataset = JoinedDataset(root=pathh, data_type='image',
                            loading_method="eager", load_labels=False)

    batch_loader = BatchLoader(dataset, batch_size=75, shuffle=False,
                               include_last_batch=True)
    print(len(batch_loader))

    for batch in batch_loader:
        print(len(batch))
        print(batch[0].size)

    print(f"Total number of batches: {len(batch_loader)}")
