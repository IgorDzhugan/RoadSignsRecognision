import numpy as np
import imutils
from pytesseract import pytesseract
import cv2
import os

PATH_TO_TESSERACT = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def find_contours(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_filter = cv2.bilateralFilter(gray_img, 11, 15, 15)
    edges = cv2.Canny(img_filter, 30, 200)
    contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return sorted(contours, key=cv2.contourArea, reverse=True)[:5]


def signs_recognision(contours):
    signs = []
    for cntrs in contours:
        approx = cv2.approxPolyDP(cntrs, 0.01 * cv2.arcLength(cntrs, True), True)
        signs.append(approx)
    return signs


def signs_crop_image(signs_coord, image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cropped = []
    for pos in signs_coord:
        mask = np.zeros(gray_img.shape, np.uint8)
        cv2.drawContours(mask, [pos], 0, 255, -1)
        (x, y) = np.where(mask == 255)
        (x1, y1) = np.min(x), np.min(y)
        (x2, y2) = np.max(x), np.max(y)
        cropped.append(gray_img[x1:x2, y1:y2])
    return cropped


def find_features(img1):
    correct_matches_dct = {}
    directory = 'TrafficSigns\\'
    for image in os.listdir(directory):
        img2 = cv2.imread(directory+image, 0)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        correct_matches = []
        for m, n in matches:
            if m.distance < 0.67*n.distance:
                correct_matches.append([m])
                correct_matches_dct[image.split('.')[0]] = len(correct_matches)
    correct_matches_dct = dict(sorted(correct_matches_dct.items(), key=lambda item: item[1], reverse=True))
    if len(list(correct_matches_dct.keys())) < 1:
        return
    return list(correct_matches_dct.keys())[0]
