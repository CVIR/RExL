from skimage import io, transform, img_as_float
import pandas as pd
import numpy as np
import json
import os

import torch
from torchvision import transforms
from torch.utils.data import Dataset

class ImageNet(Dataset):
    def __init__(self, file, min_class_id, max_class_id, transform, idx = -1):
        self.file = pd.read_csv(file)
        self.data = self.file.iloc[:, :]
        self.data = np.array(self.data)
        self.data = self.data[np.where(self.data[:,2] >= min_class_id)]
        self.data = self.data[np.where(self.data[:,2] <= max_class_id)]
        self.transform = transform
        self.image_idx = idx

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if int(self.image_idx) !=-1:
            idx = int(self.image_idx)
        row_idx = self.data[idx]
        img_path = row_idx[0]
        class_id = row_idx[2]

        X = img_as_float(io.imread(img_path, as_gray=False)).astype(np.float32)
        img_size = X.shape
        bboxes = []
        sample = {'image': X, 'bboxes': bboxes, 'labels': class_id, 'image_path': img_path}
        if self.transform:
            sample = self.transform(sample)
        sample['image_size'] = img_size
        return sample


class MSCOCO_Dataset(Dataset):
    """MSCOCO dataset."""

    def __init__(self, json_file, transform=None, idx = -1):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.DataFrame.from_dict(json.load(open(json_file)), orient='index')
        self.annotations.reset_index(level=0, inplace=True)
        self.transform = transform
        self.image_idx = idx

    def __len__(self):
        return len(self.annotations)
        # return 1

    def __getitem__(self, idx, color=True):
        # idx = 0
        # print(idx)
        if int(self.image_idx) != -1:
            idx = int(self.image_idx)
        # print(idx)
        # idx = int(idx)
        # print(self.idx)
        img_name = self.annotations['image_path'][idx]
        # image = io.imread(img_name)
        img = img_as_float(io.imread(img_name, as_gray=not color)).astype(np.float32)
        img_size = img.shape
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
            if color:
                img = np.tile(img, (1, 1, 3))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        bboxes = self.annotations['bbox_info'][idx]
        labels = self.annotations['classification_labels'][idx]
        sample = {'image': img, 'bboxes': bboxes, 'labels': labels, 'image_path': img_name}

        if self.transform:
            sample = self.transform(sample)

        sample = {'image': sample['image'], 'labels': sample['labels'], 'image_path':sample['image_path'], 'image_size': img_size, 'bboxes': sample['bboxes']}

        return sample

    def getSampleByImagePath(self, imgPath, color=True):
        record_from_json = self.annotations[self.annotations.image_path==imgPath]
        # from IPython.core.debugger import set_trace; set_trace()
        img_name = record_from_json.image_path.iloc[0] # record_from_json.image_path[0]
        # image = io.imread(img_name)
        img = img_as_float(io.imread(img_name, as_gray=not color)).astype(np.float32)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
            if color:
                img = np.tile(img, (1, 1, 3))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        bboxes = record_from_json.bbox_info.iloc[0]
        labels = record_from_json.classification_labels.iloc[0]
        sample = {'image': img, 'bboxes': bboxes, 'labels': labels, 'image_path': img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w), mode='reflect', anti_aliasing=False)

        bboxes = []
        
        if len(sample['bboxes']) != 0 :
            for annotation in sample['bboxes']:
                bbox = [0]*4
                bbox[0] = annotation['bbox'][0] * new_w / w
                bbox[1] = annotation['bbox'][1] * new_h / h
                bbox[2] = annotation['bbox'][2] * new_w / w
                bbox[3] = annotation['bbox'][3] * new_h / h

                # bboxes.append(bbox)
                bboxes.append({'bbox':bbox, 'category':annotation['category'], \
                           'category_id':annotation['category_id'], 'class_id':annotation['class_id']})

        return {'image': img, 'bboxes': bboxes, 'labels': sample['labels'], 'image_path': sample['image_path']}

class ScaleIntensities(object):
    """Convert image intensities to lie between 0 and 255."""

    def __call__(self, sample, scale=255.0):
        image, labels = sample['image'], sample['labels']

        # Multiply the intensity values
        image = image*scale

        return {'image': image,
                'labels': labels,
                'image_path': sample['image_path'], 'bboxes': sample['bboxes']}

class MakeCHWformat(object):

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # we need: C X H X W
        image = image.transpose((2, 0, 1))

        return {'image': image,
                'labels': labels,
                'image_path': sample['image_path'], 'bboxes': sample['bboxes']}


class RGBtoBGR(object):
    """Convert image from RGB to BGR."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # swap RGB to BGR
        image = image[(2,1,0), :, :] # np.flip(image,axis=0).copy() # image[:,:,::-1]

        return {'image': image,
                'labels': labels,
                'image_path': sample['image_path'], 'bboxes': sample['bboxes']}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']



        return {'image': torch.from_numpy(image.astype(float)),
                'labels': torch.from_numpy(np.asarray(labels)),
                'image_path': sample['image_path'], 'bboxes': sample['bboxes']}

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel. If None, it will not divide with std
    """

    def __init__(self, mean, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        image, labels = sample['image'], sample['labels']
        for channel in range(3):
            if self.std:
                image[channel] = (image[channel] - self.mean[channel]) / self.std[channel]
            else:
                image[channel] = image[channel] - self.mean[channel]
        return {'image': image, 'labels': sample['labels'], 'image_path': sample['image_path'], 'bboxes': sample['bboxes']}

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def load_MSCOCO_2014_json(mode,MSCOCO_Root_Path):
    return json.load(open(MSCOCO_Root_Path + 'annotations/instances_' + mode + '2014.json'))

def load_PASCAL_2012_json(mode,PASCAL_Root_Path):
    return json.load(open(PASCAL_Root_Path + 'VOC_JSON_From_COCO/pascal_' + mode + '2012.json'))

def load_PASCAL_2007_json(mode,PASCAL_Root_Path):
    return json.load(open(PASCAL_Root_Path + 'VOC_JSON_From_COCO/pascal_' + mode + '.json'))

def get_cat_id_class_id_mapping(json_data):
    categories_dict = json_data['categories']
    # print(json_data)
    running_cat_id_bias = 1
    cat_id_class_id_mapping = {}
    for i_cat in range(len(categories_dict)):
        if i_cat+running_cat_id_bias != categories_dict[i_cat]['id']:
            running_cat_id_bias = categories_dict[i_cat]['id'] - i_cat
            # print(i_cat, running_cat_id_bias)
        cat_id_class_id_mapping[categories_dict[i_cat]['id']] = {'name': categories_dict[i_cat]['name'],
                                                        'class_label': i_cat}

    return cat_id_class_id_mapping


def get_image_id_image_path_dict(json_from_COCO, MSCOCO_Root_Path, mode):
    image_id_image_path_dict = {image['id']:MSCOCO_Root_Path+mode+'2014/'+image['file_name'] \
                                for image in json_from_COCO['images']}
    return image_id_image_path_dict


def get_image_id_image_path_dict_pascal(json_from_PASCAL, PASCAL_Root_Path, mode):
    image_id_image_path_dict = {image['id']:PASCAL_Root_Path+'JPEGImages/'+image['file_name'] \
                                for image in json_from_PASCAL['images']}
    return image_id_image_path_dict


def get_annotation_dict(json_data, image_id_image_path_dict, cat_id_class_id_mapping_dict):
    print(len(cat_id_class_id_mapping_dict))
    annotation_dict = {image['id']:{'bbox_info':[], 'image_path':image_id_image_path_dict[image['id']], \
                                    'classification_labels':[0]*len(cat_id_class_id_mapping_dict)} for image in json_data['images']}

    # print(annotation_dict)
    for annotation in json_data['annotations']:
        try:
            annotation_dict[annotation['image_id']]['bbox_info'].append({'bbox':annotation['bbox'], \
                            'category_id':annotation['category_id'], \
                            'class_id':cat_id_class_id_mapping_dict[annotation['category_id']]['class_label'], \
                            # 'category':json_from_COCO['categories'][annotation['category_id']-1]['name']})
                            'category':cat_id_class_id_mapping_dict[annotation['category_id']]['name']})
        except IndexError:
            from IPython.core.debugger import set_trace; set_trace()
        # try:
        # print(annotation_dict[annotation['image_id']])
        # print(cat_id_class_id_mapping_dict)
        annotation_dict[annotation['image_id']]['classification_labels'][cat_id_class_id_mapping_dict[annotation['category_id']]['class_label']] = 1
        # except:
        #     print(annotation)
#     from IPython.core.debugger import set_trace; set_trace()
    return annotation_dict

def prep_annotation_json(MSCOCO_Root_Path, mode):
    if not os.path.isfile(MSCOCO_Root_Path + mode + '_MSCOCO_annotation.json'):
        # Load the json from COCO
        MSCOCO_2014_json = load_MSCOCO_2014_json(mode, MSCOCO_Root_Path)
        # Create a dict with image ids as keys and image paths as values
        image_id_image_path_dict_2014 = get_image_id_image_path_dict(MSCOCO_2014_json, MSCOCO_Root_Path, mode)
        # MSCOCO does not have continuous class ids. So make them continuous
        cat_id_class_id_mapping_dict = get_cat_id_class_id_mapping(MSCOCO_2014_json)
        # Create a dict with image ids as keys and bounding box infos and image paths as values
        annotation_dict = get_annotation_dict(MSCOCO_2014_json, image_id_image_path_dict_2014, cat_id_class_id_mapping_dict)

        # Save the json
        with open(MSCOCO_Root_Path + mode + '_MSCOCO_annotation.json', 'w') as outfile:
            outfile.write(json.dumps(annotation_dict,sort_keys=True, indent=2, separators=(',', ': ')))

        # Save the cat_id_class_id_mapping_dict
        with open(MSCOCO_Root_Path + mode + '_MSCOCO_cat_id_class_id_mapping.json', 'w') as outfile:
            outfile.write(json.dumps(cat_id_class_id_mapping_dict,sort_keys=True, indent=2, separators=(',', ': ')))

def prep_annotation_json_pascal(PASCAL_Root_Path, mode):
    if not os.path.isfile(PASCAL_Root_Path + mode + '_PASCAL_annotation.json'):
        # Load the json from COCO
        PASCAL_2007_json = load_PASCAL_2007_json(mode, PASCAL_Root_Path)
        # Create a dict with image ids as keys and image paths as values
        image_id_image_path_dict_2007 = get_image_id_image_path_dict_pascal(PASCAL_2007_json, PASCAL_Root_Path, mode)
        # MSCOCO does not have continuous class ids. So make them continuous
        cat_id_class_id_mapping_dict = get_cat_id_class_id_mapping(PASCAL_2007_json)
        # Create a dict with image ids as keys and bounding box infos and image paths as values
        annotation_dict = get_annotation_dict(PASCAL_2007_json, image_id_image_path_dict_2007, cat_id_class_id_mapping_dict)

        # Save the json
        with open(PASCAL_Root_Path + mode + '_PASCAL_annotation.json', 'w') as outfile:
            outfile.write(json.dumps(annotation_dict,sort_keys=True, indent=2, separators=(',', ': ')))

        # Save the cat_id_class_id_mapping_dict
        with open(PASCAL_Root_Path + mode + '_PASCAL_cat_id_class_id_mapping.json', 'w') as outfile:
            outfile.write(json.dumps(cat_id_class_id_mapping_dict,sort_keys=True, indent=2, separators=(',', ': ')))


def prep_annotation_json_pascal12(PASCAL_Root_Path, mode):
    if not os.path.isfile(PASCAL_Root_Path + mode + '_PASCAL_annotation.json'):
        # Load the json from COCO
        PASCAL_2012_json = load_PASCAL_2012_json(mode, PASCAL_Root_Path)
        # Create a dict with image ids as keys and image paths as values
        image_id_image_path_dict_2012 = get_image_id_image_path_dict_pascal(PASCAL_2012_json, PASCAL_Root_Path, mode)
        # MSCOCO does not have continuous class ids. So make them continuous
        cat_id_class_id_mapping_dict = get_cat_id_class_id_mapping(PASCAL_2012_json)
        # Create a dict with image ids as keys and bounding box infos and image paths as values
        annotation_dict = get_annotation_dict(PASCAL_2012_json, image_id_image_path_dict_2012, cat_id_class_id_mapping_dict)

        # Save the json
        with open(PASCAL_Root_Path + mode + '_PASCAL_annotation.json', 'w') as outfile:
            outfile.write(json.dumps(annotation_dict,sort_keys=True, indent=2, separators=(',', ': ')))

        # Save the cat_id_class_id_mapping_dict
        with open(PASCAL_Root_Path + mode + '_PASCAL_cat_id_class_id_mapping.json', 'w') as outfile:
            outfile.write(json.dumps(cat_id_class_id_mapping_dict,sort_keys=True, indent=2, separators=(',', ': ')))

def class_specific_annotation_json(Root_Path, dataset, mode, class_index):

    annotation = pd.DataFrame.from_dict(json.load(open(Root_Path + mode + '_'+dataset+'_annotation.json')),orient='index')
    annotation.reset_index(level=0, inplace=True)
    classification_labels = annotation['classification_labels']
    ids = []
    for idx in range(len(annotation)):
        if classification_labels[idx][class_index] == 1:
            ids.append(idx)
    filtered_annotations = annotation.iloc[ids,1:]
    annotation_dictionary = pd.DataFrame.to_dict(filtered_annotations, orient='index')

    with open(Root_Path + mode + '_class_{}_'.format(class_index)+dataset+'_annotation.json', 'w') as fp:
        json.dump(annotation_dictionary, fp, indent=4)


def getImageNet(IMAGENET_Root_Path, datasetType='train', class_id=-1,  no_of_training_classes = 100, idx=-1):
    
    tsfrm_imgnet = transforms.Compose([Rescale((224, 224)),
                                    MakeCHWformat(),
                                    ToTensor(),
                                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])
    dataset = None
    if class_id == -1:
        dataset = ImageNet(IMAGENET_Root_Path+'imagenet_'+datasetType+'.csv', 0, 999, tsfrm_imgnet, idx)
    else:
        dataset = ImageNet(IMAGENET_Root_Path+'imagenet_'+datasetType+'.csv', class_id, class_id + no_of_training_classes - 1, tsfrm_imgnet, idx)
            
    return dataset

def getPASCAL(PASCAL_Root_Path, datasetType='train', class_id=-1, idx=-1):
    prep_annotation_json_pascal(PASCAL_Root_Path, datasetType)
    if class_id != -1:
        class_specific_annotation_json(PASCAL_Root_Path, 'PASCAL', datasetType, class_id)
    
    tsfrm = transforms.Compose([Rescale((224, 224)),
                                       ScaleIntensities(),
                                       MakeCHWformat(),
                                        RGBtoBGR(),
                                        ToTensor(),
                                        Normalize([104.01, 116.67, 122.68])
                                    ])

    dataset = None
    if class_id == -1:
        dataset = MSCOCO_Dataset(json_file=PASCAL_Root_Path + datasetType + '_PASCAL_annotation.json',transform=tsfrm, idx = idx)
    else :
        dataset = MSCOCO_Dataset(json_file=PASCAL_Root_Path + datasetType + '_class_{}_PASCAL_annotation.json'.format(class_id) ,transform=tsfrm, idx = idx)
    
    return dataset

def getPASCAL12(PASCAL_Root_Path, datasetType='train', class_id=-1, idx=-1):
    prep_annotation_json_pascal12(PASCAL_Root_Path, datasetType)
    if class_id != -1:
        class_specific_annotation_json(PASCAL_Root_Path, 'PASCAL', datasetType, class_id)

    tsfrm = transforms.Compose([Rescale((224, 224)),
                                    MakeCHWformat(),
                                    ToTensor(),
                                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

    dataset = None
    if class_id == -1:
        dataset = MSCOCO_Dataset(json_file=PASCAL_Root_Path + datasetType + '_PASCAL_annotation.json',transform=tsfrm, idx = idx)
    else :
        dataset = MSCOCO_Dataset(json_file=PASCAL_Root_Path + datasetType + '_class_{}_PASCAL_annotation.json'.format(class_id) ,transform=tsfrm, idx = idx)
    
    return dataset

def getMSCOCO(MSCOCO_Root_Path, datasetType='train', class_id=-1, idx = -1):
    prep_annotation_json(MSCOCO_Root_Path, datasetType)
    if class_id != -1:
        class_specific_annotation_json(MSCOCO_Root_Path,'MSCOCO', datasetType, class_id)
    
    tsfrm = transforms.Compose([Rescale((224, 224)),
                                       ScaleIntensities(),
                                       MakeCHWformat(),
                                        RGBtoBGR(),
                                        ToTensor(),
                                        Normalize([104.01, 116.67, 122.68])
                                    ])
    dataset = None        
    # tsfrm = None                            
    if class_id == -1:
        dataset = MSCOCO_Dataset(json_file=MSCOCO_Root_Path + datasetType + '_MSCOCO_annotation.json',transform=tsfrm, idx= idx)
    else :
        dataset = MSCOCO_Dataset(json_file=MSCOCO_Root_Path + datasetType + '_class_{}_MSCOCO_annotation.json'.format(class_id) ,transform=tsfrm, idx=idx)
        
    return dataset