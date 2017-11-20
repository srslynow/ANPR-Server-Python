import numpy as np
import cv2
import os
os.environ['GLOG_minloglevel'] = '2'
#from keras.models import load_model
#import caffe
# caffe.set_mode_gpu()


class Segmentation:
    segmentation_size = (112, 512)
    #segmentation_proto = 'models/segmentation2/deploy.prototxt'
    #segmentation_model = 'models/segmentation2/model.caffemodel'
    #net = load_model("models/tf_segmentation/tensorflow_model_v1.h5")

    def segment(self, image):
        image = self.preprocess(image)
        self.preprocessed_img = image
        #image = image.astype('float32')
        #image /= 255.
        #image -= 0.5
        #image = image[np.newaxis,...,np.newaxis]
        #output = self.net.predict(image)
        #output[output < 0.95] = 0.
        #output[output >= 0.95] = 1.
        #out_img = (output * 255.).astype("uint8")
        #cv2.imshow("out", out_img[0,...])
        #cv2.waitKey(0)
        out_img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 87, 6)
        out_img = np.squeeze(out_img)
        cv2.imshow("img", out_img)
        cv2.waitKey(0)
        return out_img

    def preprocess(self, image):
        image.astype(float, copy=False)
        image = cv2.resize(image, (self.segmentation_size[1], self.segmentation_size[0]))
        return image

    def get_preprocessed_img(self):
        return self.preprocessed_img
