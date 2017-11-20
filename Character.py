import os

import numpy as np

import caffe
import cv2


class Character:
    batch_size = 12
    character_size = (28, 28)
    character_maxsize = 24
    character_proto = 'models/characters2/deploy.prototxt'
    character_model = 'models/characters2/model.caffemodel'
    label_filename = 'models/characters2/labels.txt'
    prep_imgs = []
    net = caffe.Net(character_proto, character_model, caffe.TEST)
    batch = np.empty(
        (batch_size, 1, character_size[0], character_size[1]), dtype=np.float32)

    def __init__(self):
        label_file = open(self.label_filename)
        self.labels = label_file.read().replace('\r', '').split('\n')

    def classify(self, images):
        self.prep_imgs = []
        if len(images) > self.batch_size:
            raise Exception('Too many characters to push through network')

        for index, image in enumerate(images):
            image = self.preprocess(image)
            self.prep_imgs.append(image)
            self.batch[index, 0, ...] = image
        self.net.blobs['data'].data[...] = self.batch
        self.net.forward()
        class_ids = np.argmax(self.net.blobs['softmax'].data, axis=1)
        output_str = ''
        for class_id, img_id in zip(class_ids, range(len(images))):
            if class_id < len(self.labels):
                output_str += self.labels[class_id]
            else:
                raise Exception("Predicted class id out of range")
        return output_str

    def preprocess(self, image):
        if len(image.shape) == 2:
            image = self.resize(image)
        else:
            raise Exception('Unexpected image shape')
        return image

    def resize(self, img):
        ratio = img.shape[0] / float(img.shape[1])
        if ratio < 1:
            scale = img.shape[1] / float(self.character_maxsize)
            target = int(img.shape[0] / scale)
            if target % 2 == 1:
                target = target + 1
            img = cv2.resize(img, (self.character_maxsize, target))
        else:
            scale = img.shape[0] / float(self.character_maxsize)
            target = int(img.shape[1] / scale)
            if target % 2 == 1:
                target = target + 1
            img = cv2.resize(img, (target, self.character_maxsize))
        diffH = int((self.character_size[0] - img.shape[0]) / 2)
        diffW = int((self.character_size[1] - img.shape[1]) / 2)
        img = cv2.copyMakeBorder(
            img, diffH, diffH, diffW, diffW, cv2.BORDER_CONSTANT, value=255)
        return img

    def getPrepImgs(self):
        return self.prep_imgs
