import librosa
from preprocessingABC import PreprocessingTechniqueABC
from joinedDataset import JoinedDataset
import random


class RandomAudioCrop(PreprocessingTechniqueABC):
    def __init__(self, duration):
        self.duration = duration  # Duration in seconds

    def __call__(self, audio, sr):
        track_duration = librosa.get_duration(y=audio, sr=sr)
        if track_duration <= self.duration:
            return audio, sr

        max_start = track_duration - self.duration
        start_time = random.uniform(0, max_start)
        end_time = start_time + self.duration

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        return audio[start_sample:end_sample], sr


if __name__ == "__main__":
    path = r"datasets\audio\regression\audio"
    dataset = JoinedDataset(root=path, data_type='audio',
                            loading_method="eager", load_labels=True)
    audio, label = dataset[0]
    print(audio[1])
    print(librosa.get_duration(y=audio[0], sr=audio[1]))
    randomcrop = RandomAudioCrop(5)
    cropped = randomcrop(audio[0], audio[1])
    print(librosa.get_duration(y=cropped[0], sr=cropped[1]))
