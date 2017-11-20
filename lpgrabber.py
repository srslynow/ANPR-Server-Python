from skimage import io
import random
import string
import cv2
import numpy as np

rotate_range = 10
shift_xrange = 50
shift_yrange = 10
gaussian_noise = 0.2

def disp(image):
    cv2.imshow("window", image)
    cv2.waitKey(0)

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def lp_generator(random_lp):
    dash_nr = random.randint(1,4)
    random_lp = random_lp[:dash_nr] + '-' + random_lp[dash_nr:]
    new_range = list(range(1,7))
    new_range.remove(dash_nr)
    new_range.remove(dash_nr+1)
    dash_nr = random.sample(new_range,1)
    random_lp = random_lp[:dash_nr[0]] + '-' + random_lp[dash_nr[0]:]
    return random_lp

def download_image(random_lp):
    image = io.imread('http://got.sa007.nl/kenteken/?text=' + random_lp)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (512, 112))
    return image

def add_noise(image):
    image_normalized = np.zeros(image.shape)
    noise = np.zeros(image.shape)
    cv2.normalize(image, image_normalized, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
    cv2.randn(noise, 0, random.random()/5)
    image_normalized = image_normalized + noise
    cv2.normalize(image_normalized, image_normalized, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
    cv2.normalize(image_normalized, image_normalized, 0, 255, cv2.NORM_MINMAX, cv2.CV_64F)
    image_normalized = image_normalized + random.randint(-50,100)
    image_normalized[image_normalized > 255] = 255
    image_normalized[image_normalized < 0] = 0
    image_normalized = cv2.GaussianBlur(image_normalized, (9,9), 0)
    image_normalized = image_normalized.astype(np.uint8, copy=False)
    return image_normalized

def threshold(image):
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 5)
    orig_thresh = cv2.bitwise_not(image)
    image, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    new_image = np.zeros(image.shape, np.uint8)
    #disp(image)
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        if x > 50 and w > 10 and w < 80 and h > 20 and h < 80:
            subimg = orig_thresh[y:y+h, x:x+w]
            nonz = np.count_nonzero(subimg)
            if nonz < 150:
                continue
            new_image[y:y+h, x:x+w] = orig_thresh[y:y+h, x:x+w]
    #cv2.normalize(new_image, new_image, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    image = cv2.bitwise_not(new_image)
    return image

def rotate(image, image_thresh):
    rows,cols = image.shape
    angle = random.randint(0-rotate_range,rotate_range)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    image = cv2.warpAffine(image,M,(cols,rows), borderValue=random.randint(0,255))
    image_thresh = cv2.warpAffine(image_thresh,M,(cols,rows), borderValue=255)
    return image, image_thresh

def translate(image, image_thresh):
    rows,cols = image.shape
    M = np.float32([[1,0,random.randint(0-shift_xrange, shift_xrange)],[0,1,random.randint(0-shift_yrange, shift_yrange)]])
    image = cv2.warpAffine(image,M,(cols,rows),borderValue=random.randint(0,255))
    image_thresh = cv2.warpAffine(image_thresh,M,(cols,rows),borderValue=255)
    return image, image_thresh

def getImages():
    random_lp = id_generator()
    random_lp = lp_generator(random_lp)
    image = download_image(random_lp)
    image_thresh = threshold(image)
    image, image_thresh = rotate(image, image_thresh)
    image, image_thresh = translate(image, image_thresh)
    image = add_noise(image)
    return image, image_thresh
