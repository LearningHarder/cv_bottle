import numpy as np
from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree

# files = os.listdir('./bottle/Images/')
# file_path = './bottle/Images/Image_20191006232054610.jpg'
# for f in files:
#     img = Image.open('./bottle/Images/'+f)

#     w = img.width       #图片的宽
#     h = img.height      #图片的高
#     if w!=1280 or h!= 960:
#         print(w,h)
# a,b = [1,2]
# for line in open('./bottle/Labels/Image_20191006232054610.txt'):
#     a,b,c,d,e = line.strip().split(" ")
#     print(a)
class BottleDataSet(Dataset):
    """读取解析PASCAL VOC2007/2012数据集"""

    def __init__(self, voc_root,  transforms=None, txt_name: str = "train.txt",width = 1280, height = 960):
        # assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        
        self.root = voc_root
        self.width = width
        self.height = height
        self.img_root = os.path.join(self.root, "Images")
        self.annotations_root = os.path.join(self.root, "Labels")

        # read train.txt or val.txt file
        txt_path = os.path.join(self.root,txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        with open(txt_path) as read:
            xml_list = [os.path.join(self.annotations_root, line.strip() + ".txt")
                        for line in read.readlines() if len(line.strip()) > 0]

        self.xml_list = []
        # check file
        for xml_path in xml_list:
            if os.path.exists(xml_path) is False:
                print(f"Warning: not found '{xml_path}', skip this annotation file.")
                continue

            # check for targets
            
            self.xml_list.append(xml_path)
            
        json_file = './bottle_classes.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        with open(json_file, 'r') as f:
            self.class_dict = json.load(f)

        self.transforms = transforms

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        
        # read xml
        xml_path = self.xml_list[idx]
        # print(xml_path)
        img_path = os.path.join(self.img_root, xml_path.split("/")[-1][:-4]+".jpg")
        # print(img_path)
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []
        # dw = 1. / size[0]
    # dh = 1. / size[1]
    # x = (box[0] + box[1]) / 2.0
    # y = (box[2] + box[3]) / 2.0
    # w = box[1] - box[0]
    # h = box[3] - box[2]
    # x = x * dw
    # w = w * dw
    # y = y * dh
    # h = h * dh
    # return (x, y, w, h)
        for line in open(xml_path): 
            obj_class,x,y,w,h = line.strip().split(" ")
            obj_class =int(obj_class) + 1
            x = float(x) * self.width
            y = float(y) * self.height
            w = float(w) * self.width
            h = float(h) * self.height
            B =  x + w/2
            A = x - w/2
            D = y + h /2
            C = y - h/2
            xmin, xmax, ymin, ymax = A, B, C,D
    
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj_class)
    #         if "difficult" in obj:
    #             iscrowd.append(int(obj["difficult"]))
    #         else:
            iscrowd.append(0)

    #     # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
    #     # read xml
    #     xml_path = self.xml_list[idx]
    #     with open(xml_path) as fid:
    #         xml_str = fid.read()
    #     xml = etree.fromstring(xml_str)
    #     data = self.parse_xml_to_dict(xml)["annotation"]
    #     data_height = int(data["size"]["height"])
    #     data_width = int(data["size"]["width"])
        return self.height, self.width

    # def parse_xml_to_dict(self, xml):
    #     """
    #     将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    #     Args:
    #         xml: xml tree obtained by parsing XML file contents using lxml.etree

    #     Returns:
    #         Python dictionary holding XML contents.
    #     """

    #     if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
    #         return {xml.tag: xml.text}

    #     result = {}
    #     for child in xml:
    #         child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
    #         if child.tag != 'object':
    #             result[child.tag] = child_result[child.tag]
    #         else:
    #             if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
    #                 result[child.tag] = []
    #             result[child.tag].append(child_result[child.tag])
    #     return {xml.tag: result}

    def coco_index(self, idx):
    #     """
    #     该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
    #     由于不用去读取图片，可大幅缩减统计时间

    #     Args:
    #         idx: 输入需要获取图像的索引
    #     """
    #     xml_path = self.xml_list[idx]
        # print(xml_path)
        xml_path = self.xml_list[idx]
        img_path = os.path.join(self.img_root, xml_path.split("/")[-1][:-4]+".jpg")
        # print(img_path)
        image = Image.open(img_path)
        if image.format != "JPEG":
            raise ValueError("Image '{}' format not JPEG".format(img_path))

        boxes = []
        labels = []
        iscrowd = []
        # dw = 1. / size[0]
    # dh = 1. / size[1]
    # x = (box[0] + box[1]) / 2.0
    # y = (box[2] + box[3]) / 2.0
    # w = box[1] - box[0]
    # h = box[3] - box[2]
    # x = x * dw
    # w = w * dw
    # y = y * dh
    # h = h * dh
    # return (x, y, w, h)
        for line in open(xml_path): 
            obj_class,x,y,w,h = line.strip().split(" ")
            obj_class =int(obj_class) + 1
            x = float(x) * self.width
            y = float(y) * self.height
            w = float(w) * self.width
            h = float(h) * self.height
            B =  x + w/2
            A = x - w/2
            D = y + h /2
            C = y - h/2
            xmin, xmax, ymin, ymax = A, B, C,D
    
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj_class)
    #         if "difficult" in obj:
    #             iscrowd.append(int(obj["difficult"]))
    #         else:
            iscrowd.append(0)

    #     # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (self.height, self.width), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

# import transforms
# from draw_box_utils import draw_objs
# from PIL import Image
# import json
# import matplotlib.pyplot as plt
# import torchvision.transforms as ts
# import random

# # read class_indict
# category_index = {}
# try:
#     json_file = open('./bottle_classes.json', 'r')
#     class_dict = json.load(json_file)
#     category_index = {str(v): str(k) for k, v in class_dict.items()}
# except Exception as e:
#     print(e)
#     exit(-1)

# data_transform = {
#     "train": transforms.Compose([transforms.ToTensor(),
#                                  transforms.RandomHorizontalFlip(0.5)]),
#     "val": transforms.Compose([transforms.ToTensor()])
# }
# # print(category_index)
# # load train data set
# train_data_set = BottleDataSet('./bottle/', data_transform["train"], "train.txt")
# print(len(train_data_set))
# for index in random.sample(range(0, len(train_data_set)), k=5):
#     img, target = train_data_set[index]
#     # print(img)
#     # print(target)
#     img = ts.ToPILImage()(img)
#     img =img.convert('RGB')
#     # print(img)
#     plot_img = draw_objs(img,
#                          target["boxes"].numpy(),
#                          target["labels"].numpy(),
#                          np.ones(target["labels"].shape[0]),
#                          category_index=category_index,
#                          box_thresh=0.5,
#                          line_thickness=3,
#                          font='arial.ttf',
#                          font_size=20)
#     plt.imshow(plot_img)
#     plt.show()
#     plt.savefig('bottle_img.jpg')
