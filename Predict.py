import numpy as np
import cv2
import os
import random
import string
import logging
from Finder import Finder
from Segmentation import Segmentation
from Character import Character

class Predict:
    finder = Finder()
    segmenter = Segmentation()
    charcterClassfier = Character()
    write_location = 'output/unconfirmed/'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    def predict(self, img):
        licenceplates_img = self.finder.find_lps(img)
        licenceplates = []
        for lp in licenceplates_img:
            seg_img = self.segmenter.segment(lp)
            lp = self.segmenter.get_preprocessed_img()
            seg_img, character_list = self.isolate_chars(seg_img)
            if len(character_list) >= 6 and len(character_list) < 12:
                result_text =  self.charcterClassfier.classify(character_list)
                character_list = self.charcterClassfier.getPrepImgs()
                licenceplates.append(result_text)
                self.save(img, lp, seg_img, character_list, result_text)
            else:
                self.logger.info("{} characters found in licence plate, discarding..".format(len(character_list)))
        return licenceplates
    
    def isolate_chars(self, img):
        chars = []
        orig_thresh = cv2.bitwise_not(img)
        #cv2.imshow("out2", orig_thresh)
        #cv2.waitKey(0)
        image, contours, _ = cv2.findContours(orig_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned_img = np.full_like(img, 255)
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            #print(x,y,w,h)
            if w > 10 and w < 80 and h > 40 and h < 100:
            #if w > 2 and h > 10:
                roi_img = img[y:y+h,x:x+w]
                cleaned_img[y:y+h,x:x+w] = img[y:y+h,x:x+w]
                chars.append([x, roi_img])
        chars = sorted(chars, key=lambda item: item[0]) # sort on x value
        chars = list(map(lambda item: item[1], chars)) # return only images
        return cleaned_img, chars
    
    def save(self, img, lp, seg_img, char_list, result):
        rand_str = self.id_generator() + "/"
        directory = self.write_location + rand_str
        if not os.path.exists(directory):
            os.makedirs(directory)
            cv2.imwrite(directory + "orig.png", img)
            cv2.imwrite(directory + "lp.png", lp)
            cv2.imwrite(directory + "seg.png", seg_img)
            for i,char in enumerate(char_list):
                cv2.imwrite(directory + str(i)+".png", char)
            with open(directory + "result.txt", 'w') as result_file:
                result_file.write(result)

    def id_generator(self, size=8, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for _ in range(size))