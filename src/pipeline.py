from preprocessingABC import PreprocessingTechniqueABC
from randomAudioCrop import RandomAudioCrop
from resampling import AudioResampling
from centerCrop import CenterCrop
from randomCrop import RandomCrop
from joinedDataset import JoinedDataset
import librosa
from typing import Callable, Tuple
from PIL import Image
import numpy as np


class PreprocessingPipeline(PreprocessingTechniqueABC):
    def __init__(self, *steps: Callable) -> None:
        self.steps = steps

    def __call__(self,
                 data: Image.Image | Tuple[np.ndarray, int]
                 ) -> Image.Image | Tuple[np.ndarray, int]:
        """
        Applies the preprocessing pipeline to the input data.

        Args:
            data: The input data to be preprocessed.

        Returns:
            Preprocessed data
        """
        for step in self.steps:
            data = step(data)
        return data


if __name__ == "__main__":
    path = r"datasets\audio\regression\audio"
    dataset = JoinedDataset(root=path, data_type='audio',
                            loading_method="eager", load_labels=True)

    random_crop = RandomAudioCrop(duration=10)
    resample = AudioResampling(new_sr=1500)

    pipeline = PreprocessingPipeline(random_crop, resample)
    audio, label = dataset[0]
    print(librosa.get_duration(y=audio[0], sr=audio[1]))

    for i in range(1):
        new_data = pipeline(audio)
        print(new_data)
        print(librosa.get_duration(y=new_data[0], sr=new_data[1]))

    path = r"datasets\image\regression\crowds"
    datasett = JoinedDataset(root=path, data_type='image',
                             loading_method="eager", load_labels=True)

    random_crop = RandomCrop(19, 19)
    center_crop = CenterCrop(20, 20)
    pipe = PreprocessingPipeline(center_crop, random_crop)
    point, label = datasett[0]
    print(point)
    new_point = pipe(point)
    print(new_point)
