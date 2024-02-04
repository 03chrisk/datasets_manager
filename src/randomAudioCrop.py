import librosa
from preprocessingABC import PreprocessingTechniqueABC
from joinedDataset import JoinedDataset
import random
import numpy as np
from typing import Tuple


class RandomAudioCrop(PreprocessingTechniqueABC):
    def __init__(self, duration: float) -> None:
        self.duration = duration

    def __call__(self, audio: Tuple[np.ndarray, int]) -> Tuple[np.ndarray,
                                                               int]:
        """
        Performs random cropping on the input audio.

        Args:
            audio (Tuple[np.ndarray, int]): The input audio data.
            tuple: A tuple containing the randomly cropped audio
            segment and its sampling rate.

        Returns:
            tuple (Tuple[np.ndarray, int]): A tuple containing the randomly
            cropped audio segment and its sampling rate.
        """

        audio_ts, sr = audio
        track_duration = librosa.get_duration(y=audio_ts, sr=sr)
        if track_duration <= self.duration:
            return audio_ts, sr

        max_start = track_duration - self.duration
        start_time = random.uniform(0, max_start)
        end_time = start_time + self.duration

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        return (audio_ts[start_sample:end_sample], sr)


if __name__ == "__main__":
    path = r"datasets\audio\regression\audio"
    dataset = JoinedDataset(root=path, data_type='audio',
                            loading_method="eager", load_labels=True)
    audio, label = dataset[0]
    print(audio[1])
    print(librosa.get_duration(y=audio[0], sr=audio[1]))
    randomcrop = RandomAudioCrop(5)
    cropped = randomcrop(audio)
    print(librosa.get_duration(y=cropped[0], sr=cropped[1]))
