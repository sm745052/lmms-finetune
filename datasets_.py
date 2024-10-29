import av
import os
import json
from PIL import Image
from typing import Any, Dict, List, Optional
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from chatas.code.utils.dataset import (
    Dialog,
    DialogCCData,
    Utterance,
    create_image_path_by_url,
)


TO_LOAD_IMAGE: Dict[str, bool] = {
    "llava-1.5": True,
    "llava-1.6": True,
    "llava-interleave": True,
    "llava-next-video": True,
    "llava-onevision": True,
    "qwen-vl": False,
    "phi3-v": True,
    "qwen2-vl": True,
}


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning 
    which is generalized enough to handle both images and videos.
    """

    def __init__(
        self, 
        data_path: str | DialogCCData,
        model_family_id: str,
        image_folder: Optional[str] = None,
        video_folder: Optional[str] = None,
        num_frames: int = 8,
        user_key: str = "human",
        assistant_key: str = "gpt",
    ) -> None:
        super(LazySupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            self.list_data_dict = json.load(open(data_path, "r"))
        else:
            self.list_data_dict = data_path
        self.image_folder = image_folder
        self.video_folder = video_folder
        self.num_frames = num_frames
        self.load_image = TO_LOAD_IMAGE[model_family_id]
        print("hello")
        self.user_key = user_key
        self.assistant_key = assistant_key

        self.is_text_only = [
            "image" not in source and "video" not in source
            for source in self.list_data_dict
        ]

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, List]:      
        source = self.list_data_dict[i]

        images = []
        try:
            if "image" in source:
                # here we do not do any image preprocessing but rather
                # let the processor handle everything
                # in some cases this may cause slight differences
                # but should totally be fine (e.g., official llava-1.5 does padding,
                # but llava-1.5-hf (huggingface's implementation) does not)
                if isinstance(source["image"], list):
                    image_sources = source["image"]
                elif isinstance(source["image"], str):
                    image_sources = [source["image"]]
                else:
                    raise ValueError(f"Invalid image source type: {type(source['image'])}")
                
                for image_path in image_sources:
                    if self.image_folder is not None:
                        image_path = os.path.join(self.image_folder, image_path)
                    images.append(
                        Image.open(image_path).convert("RGB")
                        if self.load_image else image_path
                    )

            videos = []
            if "video" in source:
                if isinstance(source["video"], list):
                    video_sources = source["video"]
                elif isinstance(source["video"], str):
                    video_sources = [source["video"]]
                else:
                    raise ValueError(f"Invalid video source type: {type(source['video'])}")

                num_frames = [self.num_frames] * len(video_sources)

                for video_path, cur_num_frames in zip(video_sources, num_frames):
                    if self.video_folder is not None:
                        video_path = os.path.join(self.video_folder, video_path)
                    
                    container = av.open(video_path)
                    total_frames = container.streams.video[0].frames
                    indices = np.arange(0, total_frames, total_frames / cur_num_frames).astype(int)
                    clip = read_video_pyav(container, indices)

                    videos.append(clip)
        except Exception as e:
            print(f"Error in loading image/video: {e}")
            return self.__getitem__(np.random.randint(len(self)))
        
        system_prompt = None
        if "system_prompt" in source:
            system_prompt = source["system_prompt"]

        convs = []
        assert len(source["conversations"]) > 0, "No conversations found"
        for i, conv in enumerate(source["conversations"]):
            assert conv["from"] == (self.user_key if i % 2 == 0 else self.assistant_key), "Invalid conversation"
            convs.append(conv["value"])
        assert len(convs) % 2 == 0, "Odd number of conversations"
        
        return dict(
            images=images,
            videos=videos,
            conversations=convs,
            system_prompt=system_prompt
        )


class ChatASDataset(LazySupervisedDataset):
    """
    ChatAS dataset.
    """

    def __init__(
        self,
        data_path: str,
        model_family_id: str,
        image_folder: str | None = None,
        video_folder: str | None = None,
        image_name_folder: str | None = None,
        num_frames: int = 8,
        user_key: str = "human",
        assistant_key: str = "gpt",
    ) -> None:
        image_path_by_url = create_image_path_by_url(
            image_name_folder, image_folder
        )
        raw_data = DialogCCData(
                path=data_path,
                to_filter=True,
                to_replace=True,
                image_path_by_url=image_path_by_url,
                to_unroll=False,
                min_images_per_dialog=1,
                to_split=False,
                # n_samples=100, #TODO: change to None
        )
        super().__init__(
            raw_data,
            model_family_id,
            "",
            "",
            num_frames,
            user_key,
            assistant_key,
        )
        self.is_text_only = [False] * len(self.list_data_dict)

    def transform_dialog_data_to_raw_data(
        self, dialog: Dialog, suffix: str
    ) -> List[Dict]:
        assert suffix is not None, "suffix is None"
        images = []
        for utterance in dialog.utterances:
            images.extend(utterance.images)
        return {
            "system_prompt": "Complete the following conversation",
            "image": images,
            "conversations": [
                {
                    "from": "human",
                    "value": self.format_utterance(utterance),
                }
                for utterance in dialog.utterances
            ]
            + [
                {
                    "from": "gpt",
                    "value": suffix,
                },
            ],
        }

    def format_utterance(self, utterance: Utterance) -> str:
        images_str = "<image>"*len(utterance.images)
        return f"{images_str}{utterance.text}"

    def __getitem__(self, i) -> Dict[str, List]:
        rng = np.random.RandomState(seed=42)
        conv, _ = self.list_data_dict[i]
        unrolled = conv.unroll()
        unrolled_filter_min_one_image = list(
            filter(
                lambda x: sum([len(u.images) for u in x.utterances]) >= 1,
                unrolled,
            )
        )
        if len(unrolled_filter_min_one_image) == 0:
            print(f"This shouldnot happen, couldnot find a unroll of data that has at least one image \n DialogId: {conv.idx}")
            print(len(unrolled))
            print(self.transform_dialog_data_to_raw_data(conv, "dummy"))
            return self.__getitem__(rng.randint(len(self)))
        conv = unrolled_filter_min_one_image[rng.randint(len(unrolled_filter_min_one_image))]
        splits = conv.create_splits()
        random_split = splits[rng.randint(len(splits))]
        source = self.transform_dialog_data_to_raw_data(*random_split)
        images = []
        try:
            if "image" in source:
                # here we do not do any image preprocessing but rather
                # let the processor handle everything
                # in some cases this may cause slight differences
                # but should totally be fine (e.g., official llava-1.5 does padding,
                # but llava-1.5-hf (huggingface's implementation) does not)
                if isinstance(source["image"], list):
                    image_sources = source["image"]
                elif isinstance(source["image"], str):
                    image_sources = [source["image"]]
                else:
                    raise ValueError(f"Invalid image source type: {type(source['image'])}")

                for image_path in image_sources:
                    if self.image_folder is not None:
                        image_path = os.path.join(self.image_folder, image_path)
                    images.append(
                        Image.open(image_path).convert("RGB")
                        if self.load_image else image_path
                    )

            videos = []
            if "video" in source:
                if isinstance(source["video"], list):
                    video_sources = source["video"]
                elif isinstance(source["video"], str):
                    video_sources = [source["video"]]
                else:
                    raise ValueError(f"Invalid video source type: {type(source['video'])}")

                num_frames = [self.num_frames] * len(video_sources)

                for video_path, cur_num_frames in zip(video_sources, num_frames):
                    if self.video_folder is not None:
                        video_path = os.path.join(self.video_folder, video_path)

                    container = av.open(video_path)
                    total_frames = container.streams.video[0].frames
                    indices = np.arange(0, total_frames, total_frames / cur_num_frames).astype(int)
                    clip = read_video_pyav(container, indices)

                    videos.append(clip)
        except Exception as e:
            print(f"Error in loading image/video: {e}")
            return self.__getitem__(rng.randint(len(self)))

        system_prompt = None
        if "system_prompt" in source:
            system_prompt = source["system_prompt"]

        convs = []
        assert len(source["conversations"]) > 0, "No conversations found"
        assert source["conversations"][-1]["from"] == self.assistant_key, "Last utterance should be from assistant"
        assert set([conv["from"] for conv in source["conversations"][:-1]]) == set([self.user_key]), "All but last utterance should be from user"
        for i, conv in enumerate(source["conversations"]):
            convs.append(conv["value"])
        # assert len(convs) % 2 == 0, "Odd number of conversations"

        return dict(
            images=images,
            videos=videos,
            conversations=convs,
            system_prompt=system_prompt
        )


if __name__ == '__main__':
    image_name_path = "../../tmp/image_names"
    image_path = "../../tmp/images_n"
    data_path = "data/DialogCC/test.csv"
    chatas_dataset = ChatASDataset(
        data_path=data_path,
        model_family_id="llava-interleave",
        image_folder=image_path,
        image_name_folder=image_name_path,
        num_frames=8,
        user_key="human",
        assistant_key="gpt",
    )
    print(chatas_dataset[20])
