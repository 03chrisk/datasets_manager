from preprocessingABC import PreprocessingTechniqueABC
from randomAudioCrop import RandomAudioCrop
from resampling import AudioResampling
from joinedDataset import JoinedDataset
import librosa


class PreprocessingPipeline(PreprocessingTechniqueABC):
    def __init__(self, *steps):
        self.steps = steps

    def __call__(self, data, sr=None):
        for step in self.steps:
            # Check if the preprocessing step requires a sampling rate
            if 'sr' in step.__call__.__code__.co_varnames:
                data, sr = step(data, sr)
            else:
                data = step(data)
        return data if sr is None else (data, sr)


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
        new_data = pipeline(audio[0], audio[1])
        print(new_data)
        print(librosa.get_duration(y=new_data[0], sr=new_data[1]))
