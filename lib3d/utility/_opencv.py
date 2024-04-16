import cv2
import numpy as np
import os


def uniquecols(n_cols=100, normalized=False):
    """
    Returns greater than n unique colors which are distinguishable in color space
    :param n_cols: number of color rows required in array
    :param normalized: if True , returns color in normalized form.
    :return: RGB colors in [N,3] shape
    """
    rnge_ = np.arange(0, 255, 255 / n_cols)
    cols = np.vstack((rnge_, np.random.randint(0, 255, rnge_.shape[0]), rnge_[::-1])).T
    if normalized:
        return cols/255.
    return cols.astype(np.uint8)


def sharpenedge(image, sigma =1 / 3):
    # image = cv2.cvtColor(image , cv2.COLOR_RGB2GRAY)
    image = cv2.GaussianBlur(image , (3,3) , 0)
    v = np.median(image)
    l_, u_ = int(max(0, (1.0 - sigma) * v)), int(min(255, (1.0 + sigma) * v))
    edge = cv2.Canny(image , l_ , u_)
    contours = np.zeros_like(image)
    cntrs, hier = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    colors = uniquecols(n_cols=len(cntrs))
    for idx, c in enumerate(cntrs):
        area = cv2.contourArea(c)
        if area < 10:continue
        contours = putconts(image=contours ,color=colors[idx] , contours=c , filled=True)
    return edge , contours

def pooling(image, mode='max', kernel=(2, 2), stride=1):
    outh, outw = image.shape[0] // kernel[0], image.shape[1] // kernel[1]
    n_dimm = image.ndim
    if mode == 'max':
        pooled = np.full(shape=(outh, outw), fill_value=-np.inf, dtype=image.dtype)
        np.maximum.at(pooled, (np.arange(outh)[:, None], np.arange(outw)), image[::kernel[0], ::kernel[1]])
    elif mode == 'min':
        pooled = np.full(shape=(outh, outw), fill_value=np.inf, dtype=image.dtype)
        np.minimum.at(pooled, (np.arange(outh)[:, None], np.arange(outw)), image[::kernel[0], ::kernel[1]])
    elif mode == 'mean':
        pooled = np.zeros(shape=(outh, outw), dtype=image.dtype)
        np.add.at(pooled, (np.arange(outh)[:, None], np.arange(outw)), image[::kernel[0], ::kernel[1]])
    else:
        # skipping pixel at kernal steps
        pooled = image[::kernel[0], ::kernel[1]]
    return pooled


def depth_normalmap(depth):
    rows, cols = depth.shape
    # Calculate the partial derivatives of depth with respect to x and y
    """
    #define CV_8U   0
    #define CV_8S   1
    #define CV_16U  2
    #define CV_16S  3
    #define CV_32S  4
    #define CV_32F  5
    #define CV_64F  6
    #define CV_16F  7
    """
    dx = cv2.Sobel(depth, 5, dx=1, dy=0, ksize=1)
    dy = cv2.Sobel(depth, ddepth=5, dx=0, dy=1, ksize=1)

    # Compute the normal vector for each pixel
    normal = np.dstack((-dx, -dy, np.ones_like(depth)))
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)

    # Map the normal vectors to the [0, 255] range and convert to uint8
    normalcolr = (normal + 1) / 2 * 255.
    normalcolr = normalcolr.clip(0, 255).astype(np.uint8)
    # normal_bgr = cv2.cvtColor(normalcolr, cv2.COLOR_RGB2BGR)
    return normal, normalcolr[..., ::-1]


def save_(image, path, filename='1', ext='.png'):
    """
    writes image in the output path
    :param image: image in np.uint8 format
    :param outputh: path to save the image
    :return: None
    """
    assert image.ndim in (1, 2, 3)
    assert image.dtype in [np.uint8, np.float32, np.float64]
    if image.dtype == np.float32 or image.dtype == np.float64:
        min_, max_ = image.min(), image.max()
        image = np.round(image * 255).astype(np.uint8)
    return cv2.imwrite(path + os.sep + filename + ext, image)


def putconts(image, contours, color, minrect=False, filled=True):
    """
    draw contours on image with specified color
    :param image: image to draw
    :param contours: points in [[pixelx,pixely]]
    :param color: color of contour
    :param filled: if True , returns filled contour else returns edges
    :return: None
    """
    if minrect:
        obox = cv2.minAreaRect(contours.astype(np.int32))
        contours = cv2.boxPoints(obox)
    if filled:
        return cv2.fillPoly(img=image, pts=[contours.astype(np.int32)], color=color.tolist())
    else:
        return cv2.polylines(img=image, pts=[contours.astype(np.int32)], isClosed=True, color=color.tolist(),
                             thickness=2)


def imempty(shape, col='w'):
    fill = [0, 0, 0]
    if col == 'w': fill = [255, 255, 255]
    if col == 'r': fill = [255, 0, 0]
    if col == 'g': fill = [0, 255, 255]
    if col == 'b': fill = [0, 0, 255]
    return np.full(shape=shape, fill_value=fill, dtype=np.uint8)


def showim(image, windowname='output', waitkey=1, dest=False):
    """
    image vizualizer
    """
    cv2.namedWindow(winname=windowname, flags=cv2.WINDOW_NORMAL)
    cv2.imshow(windowname, image)
    cv2.waitKey(waitkey)
    if dest: cv2.destroyWindow(winname=windowname)
    return


def puttext(image, text, pos=(30, 30), color=(255, 0, 0)):
    """write texts on image"""
    thickness = 2 if image.shape[0] > 1080 else 1
    fontscale = 1 if image.shape[0] > 1080 else 0.5
    return cv2.putText(img=image, text=str(text), org=pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontscale,
                       thickness=thickness, color=color, lineType=cv2.LINE_AA)


def putbbox(image, boxlist, mode='xyxy', color=(255, 0, 0)):
    """put bounding box in image"""
    thickness = 2 if image.shape[0] > 1080 else 1
    for (x, y, x1, y1) in boxlist:
        if mode in ['xywh']: cv2.rectangle(img=image, pt1=(int(x), int(y)), pt2=(int(x + x1), int(y + y1)), color=color,
                                           thickness=thickness)
        if mode in ['xyxy']: cv2.rectangle(img=image, pt1=(int(x), int(y)), pt2=(int(x1), int(y1)), color=color,
                                           thickness=thickness)
    return image


def readim(imagepath, mode=None):
    return cv2.imread(imagepath)
