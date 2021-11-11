import os
import csv 
import cv2 
import numpy as np	
from skimage import io, transform, img_as_float

def get_labels(filename):
    file = open(filename, 'r+')
    lines = file.readlines()
    labels = []
    for line in lines:
        words = line.split(':')
        labels.append(words[0])
    return labels
    
def create_annotations_file():
    labels = get_labels('imagenet_metadata.txt')
    base_path = os.getcwd()
    base_path = os.path.join(base_path, 'IMAGENET_TRAIN')
    # folder_path = os.path.join(base_path)
    ann_file = 'imagenet_train.csv'
    file = open(ann_file, 'w')
    writer = csv.writer(file)
    cl = 0
    for label in labels:
        path = os.path.join(base_path, label)
        flag = os.path.exists(path)
        if flag is False:
            cl += 1
            continue
        for img_path in os.listdir(path):
            row = [os.path.join(path, img_path), label, cl]
            img_path = row[0]
            im = img_as_float(io.imread(img_path, as_grey=False)).astype(np.float32)
            if im.ndim == 3:
               writer.writerow(row)
            else:
               print(img_path)
        cl += 1

def create_test_set(folder):
    test_labels = [0, 4, 8, 11, 13]
    labels = get_labels('wnids.txt')
    print(labels)
    base_path = os.getcwd()
    folder_path = os.path.join(base_path, folder)
    ann_file = 'imagenet_' + str(folder) + 'test.csv'
    file = open(ann_file, 'w')
    writer = csv.writer(file)
    for id in test_labels:
        label = labels[id]
        print(label )
        path = os.path.join(folder_path, label + '/images')
        for img_path in os.listdir(path):
            row = [os.path.join(path, img_path), label, id]
            writer.writerow(row)


if __name__=='__main__':
    create_annotations_file()


    
