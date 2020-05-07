from torch.utils.data import Dataset
import cv2
import torch
import os


class ImageDataset(Dataset):
    def __init__(self, root_dir, class_file, transforms=None):
        self.root_dir = root_dir
        self.ClassInfo = self._GetClassInfo(class_file)
        self.ImageInfo = self._BuildImageInfo(root_dir)
        self.transforms = transforms

    def __len__(self):
        assert len(self.ImageInfo["path"]) == len(self.ImageInfo["label"]), "# path != # label"
        return len(self.ImageInfo["path"])

    def __getitem__(self, ix):
        img_path = self.ImageInfo["path"][ix]
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]  # bgr to rgb
        label = torch.tensor(self.ImageInfo["label"][ix], dtype=torch.int64)
        if self.transforms:
            img1 = self.transforms(img)
            img2 = self.transforms(img)

        sample1 = {"image": img1, "label": label}
        sample2 = {"image": img2, "label": label}

        return (sample1, sample2)

    def _BuildImageInfo(self, Dir):
        imgs = {"path": [], "label": []}  # all paths and labels
        for folder in os.listdir(Dir):
            for file in os.listdir(Dir + "/" + folder):
                imgs["path"].append(Dir + "/" + folder + "/" + file)
                imgs["label"].append(self.ClassInfo[folder])
        return imgs

    def _GetClassInfo(self, file):
        ClassInfo = {}
        with open(file, 'r') as f:
            chars = f.read().split("\n")[:-1]
            for char in chars:
                if char != '':
                    c, L = char.split(" ")[0], int(char.split(" ")[1])
                    ClassInfo[c] = L
        return ClassInfo
