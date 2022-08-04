import cv2
from Recognision_function import *


def main():
    orig_img = cv2.imread(input('Enter filepath: '))
    signs = signs_crop_image(signs_recognision(find_contours(orig_img)), orig_img)
    signs_names = []
    for sign in signs:
        if find_features((sign)):
            signs_names.append(find_features(sign))
    print(set(signs_names))
    return


if __name__ == '__main__':
    main()
