import torch
import torchvision.transforms.functional as TF

class ToTensor(object):
    def __call__(self, sample):
        sample["image"] = TF.to_tensor(sample["image"])
        sample["label"] = torch.FloatTensor(sample["label"])
        return sample

class Normalize(object):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), from_image=False):
        """
        If 'from_image' argument is used, the 'mean' and 'std' arguments will be ignored
        and the mean, std values will be calculated from the image in sample.
        """ 
        super().__init__()
        self.mean = mean
        self.std = std
        self.from_image = from_image

    def __call__(self, sample):
        if self.from_image:
            self.mean = sample["image"].mean(dim=(1, 2))
            self.std = sample["image"].std(dim=(1, 2))
        sample["image"] = TF.normalize(sample["image"], self.mean, self.std)
        return sample