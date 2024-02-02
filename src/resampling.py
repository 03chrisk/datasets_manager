import librosa
from preprocessingABC import PreprocessingTechniqueABC
from joinedDataset import JoinedDataset


class AudioResampling(PreprocessingTechniqueABC):
    def __init__(self, new_sr):
        self.new_sr = new_sr

    def __call__(self, audio, sr):
        return librosa.resample(audio,
                                orig_sr=sr, target_sr=self.new_sr), self.new_sr


if __name__ == "__main__":
    path = r"datasets\audio\regression\audio"
    dataset = JoinedDataset(root=path, data_type='audio',
                            loading_method="eager", load_labels=True)

    audio, label = dataset[0]
    print(audio[0])
    print(audio[1])
    print(librosa.get_duration(y=audio[0], sr=audio[1]))
    resample = AudioResampling(15000)
    resampled_audio, new_sr = resample(audio[0], audio[1])

    print(resampled_audio)
    print(new_sr)
    print(librosa.get_duration(y=resampled_audio, sr=new_sr))
