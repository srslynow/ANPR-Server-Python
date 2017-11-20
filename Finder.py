import numpy as np
import cv2
import logging

class Finder:
    cascade_path = 'models/finder/cascade.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    def find_lps(self, image):
        licenceplates_rect = self.cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=15, minSize=(75,15), maxSize=(512,112))
        licenceplates = []
        for (x,y,w,h) in licenceplates_rect:
            licenceplates.append(image[y:y+h, x:x+w])
        self.logger.info("Found %i possible licenceplates" % len(licenceplates))
        return licenceplates
