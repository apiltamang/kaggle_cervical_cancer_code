import numpy as np
import cv2
import math
from glob import glob
import os

def get_filename(image_id, image_type, DATA_ROOT):
    """
    Method to get image file path from its id and type
    """
    data_path = os.path.join(DATA_ROOT, image_type)

    ext = 'jpg'
    return os.path.join(data_path, "{}.{}".format(image_id, ext))


def get_image_data(image_id, image_type, DATA_ROOT):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type, DATA_ROOT)
    img = cv2.imread(fname)
    if img is None:
        print("Failed to read image : %s, %s, %s" % (image_id, image_type, DATA_ROOT))

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def maxHist(hist):
    maxArea = (0, 0, 0)
    height = []
    position = []
    for i in range(len(hist)):
        if (len(height) == 0):
            if (hist[i] > 0):
                height.append(hist[i])
                position.append(i)
        else:
            if (hist[i] > height[-1]):
                height.append(hist[i])
                position.append(i)
            elif (hist[i] < height[-1]):
                while (height[-1] > hist[i]):
                    maxHeight = height.pop()
                    area = maxHeight * (i - position[-1])
                    if (area > maxArea[0]):
                        maxArea = (area, position[-1], i)
                    last_position = position.pop()
                    if (len(height) == 0):
                        break
                position.append(last_position)
                if (len(height) == 0):
                    height.append(hist[i])
                elif (height[-1] < hist[i]):
                    height.append(hist[i])
                else:
                    position.pop()
    while (len(height) > 0):
        maxHeight = height.pop()
        last_position = position.pop()
        area = maxHeight * (len(hist) - last_position)
        if (area > maxArea[0]):
            maxArea = (area, len(hist), last_position)
    return maxArea


def maxRect(img):
    maxArea = (0, 0, 0)
    addMat = np.zeros(img.shape)
    for r in range(img.shape[0]):
        if r == 0:
            addMat[r] = img[r]
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
        else:
            addMat[r] = img[r] + addMat[r - 1]
            addMat[r][img[r] == 0] *= 0
            area = maxHist(addMat[r])
            if area[0] > maxArea[0]:
                maxArea = area + (r,)
    return (
    int(maxArea[3] + 1 - maxArea[0] / abs(maxArea[1] - maxArea[2])), maxArea[2], maxArea[3], maxArea[1], maxArea[0])


def cropCircle(img):
    dim_threshold = 1024
    if (img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1] * dim_threshold / img.shape[0]), dim_threshold)
    else:
        tile_size = (dim_threshold, int(img.shape[0] * dim_threshold / img.shape[1]))

    img = cv2.resize(img, dsize=tile_size)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY);
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    main_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    ff = np.zeros((gray.shape[0], gray.shape[1]), 'uint8')
    cv2.drawContours(ff, main_contour, -1, 1, 15)
    ff_mask = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), 'uint8')
    cv2.floodFill(ff, ff_mask, (int(gray.shape[1] / 2), int(gray.shape[0] / 2)), 1)
    # cv2.circle(ff, (int(gray.shape[1]/2), int(gray.shape[0]/2)), 3, 3, -1)

    rect = maxRect(ff)
    img_crop = img[min(rect[0], rect[2]):max(rect[0], rect[2]), min(rect[1], rect[3]):max(rect[1], rect[3])]
    cv2.rectangle(ff, (min(rect[1], rect[3]), min(rect[0], rect[2])), (max(rect[1], rect[3]), max(rect[0], rect[2])), 3,
                  2)

    # plt.subplot(121)
    # plt.imshow(img)
    # plt.subplot(122)
    # plt.imshow(ff)
    # plt.show()

    return img_crop

def Ra_space(img, Ra_ratio, a_threshold):
    imgLab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB);
    w = img.shape[0]
    h = img.shape[1]
    Ra = np.zeros((w * h, 2))
    for i in range(w):
        for j in range(h):
            R = math.sqrt((w / 2 - i) * (w / 2 - i) + (h / 2 - j) * (h / 2 - j))
            Ra[i * h + j, 0] = R
            Ra[i * h + j, 1] = min(imgLab[i][j][1], a_threshold)

    Ra[:, 0] /= max(Ra[:, 0])
    Ra[:, 0] *= Ra_ratio
    Ra[:, 1] /= max(Ra[:, 1])

    return Ra
