import os
import pathlib

import matplotlib as mpl
import numpy as np
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

import config as cf


Image.MAX_IMAGE_PIXELS = None  # DecompressionBombError


class CodeDataset(Dataset):
    def __init__(self, root_path, img_type, db_name, partition, top_labels=None):
        self.ids, self.labels = [], {}
        lookup_path = "{}/{}/{}/{}".format(root_path, img_type, db_name, partition)
        for file in pathlib.Path(lookup_path).glob('**/*{}'.format(cf.IMG_EXT)):
            if os.path.isfile(file):
                cur_path = str(file)
                self.ids.append(cur_path)
                cur_label = cur_path.split('_')[-1].replace(cf.IMG_EXT, '')
                self.labels[cur_path] = cur_label

        if not top_labels:
            self.top_labels = sorted(list(set(self.labels.values())))
        else:
            self.top_labels = sorted(top_labels)
        self.label2index = {m: i for i, m in enumerate(self.top_labels)}
        self.index2label = {i: m for i, m in enumerate(self.top_labels)}
        for k, v in self.labels.items():
            self.labels[k] = self.label2index[v]

        self.transform = transforms.Compose([
            transforms.Resize(size=(cf.IMG_TRANS_SIZE, cf.IMG_TRANS_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        input_x = Image.open(img_id).convert(cf.IMG_MODE)
        input_x = self.transform(input_x)
        output_y = self.labels[img_id]
        return img_id, input_x, output_y

    def get_top_labels(self):
        return self.top_labels


def img_show(img):
    std_img = img / 2 + 0.5  # ((img * std) + mean)
    plt.imshow(np.transpose(std_img.numpy(), (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    test_set = CodeDataset(cf.ROOT_PATH, cf.IMG_TYPES[cf.IMG_TYPE], cf.DB_NAMES[cf.DB_NAME], cf.PARTITIONS["test"])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=cf.NUM_WORKER)
    print("test_set = #{}".format(len(test_set)))

    # get some random images
    dataiter = iter(test_loader)
    ids, images, labels = dataiter.next()
    print(ids)
    print(images.shape)
    print(labels)

    # show image
    mpl.rcParams['figure.dpi'] = 600
    img_show(torchvision.utils.make_grid(images))
