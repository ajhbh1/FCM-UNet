import os
from torch.utils.data import Dataset
import cv2
import glob
import random
from torchvision import transforms

data_transform = {
    "train": transforms.Compose([transforms.ToTensor()]),
    "test": transforms.Compose([transforms.ToTensor()])
}

class He_Loader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.bmp'))

    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        label_path = image_path.replace('image', 'label')

        # 读取
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        if label.max() > 1:
            label = label / 255
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label

    def __len__(self):
        return len(self.imgs_path)





