import os
import sys
sys.path.append(os.getcwd() + "/src/")
from src.joinedDataset import JoinedDataset  # noqa: E402
from src.treeDataset import TreeDataset  # noqa: E402
from src.batchLoader import BatchLoader  # noqa: E402
from src.pipeline import PreprocessingPipeline  # noqa: E402
from src.randomCrop import RandomCrop  # noqa: E402
from src. randomAudioCrop import RandomAudioCrop  # noqa: E402
from src.centerCrop import CenterCrop  # noqa: E402
from src.resampling import AudioResampling  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import sounddevice as sd  # noqa: E402


def main():
    audio_path = r"datasets\audio\regression\audio"
    image_path = r"datasets\image\classification"
    image_reg_path = r"datasets\image\regression\crowds"

    image_dataset = TreeDataset(root=image_path,
                                data_type='image',
                                loading_method='lazy')

    print(f"The image dataset has {len(image_dataset)} datapoints")

    audio_dataset = JoinedDataset(root=audio_path,
                                  data_type='audio',
                                  loading_method="eager",
                                  load_labels=True)

    print(f"The audio dataset has {len(audio_dataset)} datapoints")

    image_reg_dataset = JoinedDataset(root=image_reg_path,
                                      data_type='image',
                                      loading_method='eager',
                                      load_labels=False)

    print(f"The image regression dataset has {len(image_reg_dataset)}"
          " datapoints")
    print(" ")

    train, test = image_dataset.split(0.7)
    print(f"train is {len(train)} points , and test is {len(test)}")
    print(" ")

    image_batch_loader = BatchLoader(image_dataset,
                                     batch_size=75,
                                     shuffle=True,
                                     include_last_batch=True)

    print(
        f"batch size for the image dataset is {image_batch_loader.batch_size}")
    print(f"There are {len(image_batch_loader)} batches of images")
    print(" ")

    audio_batch_loader = BatchLoader(audio_dataset,
                                     batch_size=100,
                                     shuffle=False,
                                     include_last_batch=False)

    print(
        f"batch size for the audio dataset is {audio_batch_loader.batch_size}")
    print(f"There are {len(audio_batch_loader)} batches of audio")
    print(" ")

    random_image_crop = RandomCrop(50, 50)
    center_crop = CenterCrop(110, 110)
    image_pipeline = PreprocessingPipeline(center_crop, random_image_crop)

    random_audio_crop = RandomAudioCrop(duration=5)
    resample = AudioResampling(new_sr=10000)
    audio_pipeline = PreprocessingPipeline(random_audio_crop, resample)

    for batch in image_batch_loader:
        for datapoint in batch:
            data, label = datapoint
            plt.imshow(data)
            plt.axis("off")
            plt.show()
            data = image_pipeline(data)
            plt.imshow(data)
            plt.axis("off")
            plt.show()
            break

    input("you are about to hear a sound, press enter when you are ready")

    for i in range(3):
        data, label = audio_dataset[i]
        newdata = audio_pipeline(data)
        sd.play(newdata[0], newdata[1])
        sd.wait()
        print(f"played sound {i+1}, it was a {label}")

    image_reg_datapoint = image_reg_dataset[0]
    plt.imshow(image_reg_datapoint)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
