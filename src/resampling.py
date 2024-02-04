import librosa
from preprocessingABC import PreprocessingTechniqueABC
from joinedDataset import JoinedDataset
import numpy as np
from typing import Tuple


class AudioResampling(PreprocessingTechniqueABC):
    def __init__(self, new_sr: int) -> None:
        self.new_sr = new_sr

    def __call__(self, audio: Tuple[np.ndarray, int]) -> Tuple[np.ndarray,
                                                               int]:
        """
        Resamples the input audio data to the new sampling rate.

        Args:
            audio (Tuple[np.ndarray, int]): The input audio data.
            tuple: A tuple containing the randomly cropped audio
            segment and its sampling rate.

        Returns:
            tuple (Tuple[np.ndarray, int]): A tuple containing the randomly
            cropped audio segment and its sampling rate.
        """
        audio, sr = audio
        return (librosa.resample(audio,
                                 orig_sr=sr,
                                 target_sr=self.new_sr), self.new_sr)


if __name__ == "__main__":
    path = r"datasets\audio\regression\audio"
    dataset = JoinedDataset(root=path, data_type='audio',
                            loading_method="eager", load_labels=True)

    audio, label = dataset[0]
    print(audio[0])
    print(audio[1])
    print(librosa.get_duration(y=audio[0], sr=audio[1]))
    resample = AudioResampling(15000)
    resampled_audio, new_sr = resample(audio)

    print(resampled_audio)
    print(new_sr)
    print(librosa.get_duration(y=resampled_audio, sr=new_sr))
