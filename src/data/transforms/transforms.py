from torchvision import transforms


class make_grayscale_3channels:
    @staticmethod
    def make_grayscale_3channels(img):
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img

    def get(self):
        return transforms.Lambda(self.make_grayscale_3channels)


class target_class_mapping:
    @staticmethod
    def target_class_mapping(target, class_mapping):
        return class_mapping[target]

    def __init__(self, class_mapping):
        self.class_mapping = class_mapping

    def get(self):
        return transforms.Lambda(lambda x: self.target_class_mapping(x, self.class_mapping))
