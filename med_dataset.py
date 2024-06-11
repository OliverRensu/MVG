import os

import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
class MVGDataset(Dataset):
    def __init__(self, path, masked_position_generator, half_mask_ratio, seg_only=True):
        self.masked_position_generator = masked_position_generator
        self.half_mask_ratio=half_mask_ratio
        self.path = path
        self.seg_only = seg_only
        self.task = ["ACDC", "LA", "LungCT", "amosct", "amosmr", "btcv", "promise", "prostate_ultrasound", "word"]
        self.task2count={self.task[i]:i for i in range(len(self.task))}
        self.inpainting = ["brainLocal"]
        self.transfer = ["brainGLI"]
        self.file=[]
        self.seg = []
        self.inpaint=[]
        self.trans=[]
        for i in self.task:
            case = sorted(os.listdir(os.path.join(path, i, "image")))
            for c in case:
                file = sorted(os.listdir(os.path.join(path, i, "image", c)))
                for f in file:
                    self.seg.append(["seg", [i, "image", c, f], [i, "label", c, f]])
        for i in self.inpainting:
            case = sorted(os.listdir(os.path.join(path, i, "image")))
            case = case[:len(case) // 2]
            for c in case:
                file = sorted(os.listdir(os.path.join(path, i, "image", c)))
                for f in file:
                    self.inpaint.append(["inpainting", [i, "image", c, f], [i, "label", c, f]])
        for i in self.transfer:
            modality = ['t1c', 't2w']
            case = sorted(os.listdir(os.path.join(path, i, modality[0])))
            case = case[:: 4] # to balance
            for c in case:
                file = sorted(os.listdir(os.path.join(path, i, modality[0], c)))
                for f in file:
                    for m1 in modality:
                        for m2 in modality:
                            if m1!=m2:
                                self.trans.append(["transfer", [i, m1, c, f], [i, m2, c, f]])
        if self.seg_only:
            self.file=self.seg
        else:
            self.file=self.seg*2+self.inpaint+self.trans
        print("Segmentation: ", len(self.seg), "inpainting", len(self.inpaint), "transfer", len(self.trans))

        self.img_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.label_transform1 = transforms.ToTensor()
        self.label_transform2 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.file)
    def __getitem__(self, item):
        R, G, B = np.random.randint(1,11),np.random.randint(1,11),np.random.randint(1,11)
        task, image_name, label_name = self.file[item]
        img_path = os.path.join(self.path, image_name[0], image_name[1], image_name[2], image_name[3])
        label_path = os.path.join(self.path, label_name[0], label_name[1], label_name[2], label_name[3])
        images = []
        labels = []
        image = Image.open(img_path)
        image = np.array(image)
        label = Image.open(label_path)
        label = np.array(label)
        if len(image.shape)==2:
            image = image[:,:,None].repeat(3, axis=2)
        if len(label.shape)==2:
            label = label[:,:,None].repeat(3, axis=2)
        image = self.img_transform(image)
        if task=="inpainting" or task=="transfer":
            label = self.img_transform(label)
        elif task=="seg":
            label = self.label_transform1(label)*10
            label = self.label_transform2(label)
        images.append(image)
        labels.append(label)
        files = os.listdir(os.path.join(self.path, image_name[0], image_name[1], image_name[2]))
        choose = np.random.choice(len(files), [2], replace=False)
        if files[choose[0]]!=image_name[3]:
            file = files[choose[0]]
        else:
            file = files[choose[1]]
        img_path = os.path.join(self.path, image_name[0], image_name[1], image_name[2], file)
        label_path = os.path.join(self.path, label_name[0], label_name[1], label_name[2], file)
        image = Image.open(img_path)
        image = np.array(image)
        label = Image.open(label_path)
        label = np.array(label)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        if len(label.shape) == 2:
            label = label[:, :, None].repeat(3, axis=2)
        image = self.img_transform(image)
        if task=="inpainting" or task=="transfer":
            label = self.img_transform(label)
        elif task=="seg":
            label = self.label_transform1(label) * 10
            label = self.label_transform2(label)

        images.append(image)
        labels.append(label)
        left = np.random.randint(0,512-448, [2])
        up = np.random.randint(0, 512 - 448, [2])
        images = [images[0][:, left[0]:left[0]+448, up[0]:up[0]+448], images[1][:, left[1]:left[1]+448, up[1]:up[1]+448]]
        labels = [labels[0][:, left[0]:left[0] + 448, up[0]:up[0] + 448], labels[1][:, left[1]:left[1] + 448, up[1]:up[1] + 448]]

        image = torch.cat(images, dim=1)
        label = torch.cat(labels, dim=1)
        use_half_mask = torch.rand(1)[0] < self.half_mask_ratio
        if task=="seg":
            num_patches = self.masked_position_generator.num_patches
            mask = np.zeros(self.masked_position_generator.get_shape(), dtype=np.int32)
            mask[mask.shape[0] // 2:, :] = 1
        else:
            if use_half_mask:
                num_patches = self.masked_position_generator.num_patches
                mask = np.zeros(self.masked_position_generator.get_shape(), dtype=np.int32)
                mask[mask.shape[0] // 2:, :] = 1
            else:
                mask = self.masked_position_generator()
        valid = torch.ones_like(label)
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
        imagenet_std = torch.tensor([0.229, 0.224, 0.225])
        thres = torch.ones(3) * (1e-3 * 0.1)
        thres = (thres - imagenet_mean) / imagenet_std
        if task=="inpainting" or task=="transfer":
            valid[label < thres[:, None, None]] = 1
        elif task=="seg":
            valid[label < thres[:, None, None]] = 0.1
        return {"image": image, "label": label, "mask": mask, "valid":valid}

