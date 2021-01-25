import cv2


def resize(img,size=(224,224)):
    new_img = cv2.resize(img, size, interpolation = cv2.INTER_AREA)
    return new_img