import librosa
from preprocessingABC import PreprocessingTechniqueABC
import numpy as np
from typing import Tuple


class AudioResampling(PreprocessingTechniqueABC):
    def __init__(self, new_sr: int) -> None:
        if not isinstance(new_sr, int):
            raise TypeError("new_sr must be an integer")
        if new_sr <= 0:
            raise ValueError("new_sr must be a positive integer")
        self._new_sr = new_sr

    @property
    def new_sr(self):
        return self._new_sr

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
        if not isinstance(audio, tuple):
            raise TypeError("Audio must be a tuple (np.ndarray, int)")

        audio_ts, sr = audio

        if not isinstance(audio_ts, np.ndarray) or not isinstance(sr, int):
            raise ValueError("Tuple needs to consist of an np.ndarray and int")

        return (librosa.resample(audio_ts,
                                 orig_sr=sr,
                                 target_sr=self.new_sr), self.new_sr)
