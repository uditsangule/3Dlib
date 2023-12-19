import cv2
import numpy as np


def showImage(image , waitkey=0 , windowname='output'):
    """
    This Function shows image in a specified windowname
    :param image: image in (h,w) or (h,w,c) uint8
    :param waitkey: delay in holding the window, default is 0 to hold it until quit window
    :param windowname: name of the window
    :return: None
    """
    cv2.namedWindow(windowname, cv2.WINDOW_NORMAL)
    cv2.imshow(windowname, image)
    cv2.waitKey(waitkey)
    cv2.destroyWindow(windowname)
    return

def streamvideo(video , windowname='streaming' , fps=None):
    """
    This function streams the video
    :param video:
    :param windowname:
    :param fps:
    :return:
    """
    return



