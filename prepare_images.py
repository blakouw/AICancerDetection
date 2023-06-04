import glob
import random
import cv2
import numpy as np


def read_and_prepare():
    images = glob.glob(r'[path_to_folder]\archive\**\*.png', recursive=True)

    nie_rak = []
    rak = []

    for img in images:
        if img[-5] == '0':
            nie_rak.append(img)
        elif img[-5] == '1':
            rak.append(img)

    zdj_bez_raka = []
    zdj_z_rakiem = []

    for img in nie_rak:
        nie_rak_img = cv2.imread(img, cv2.IMREAD_COLOR)
        rak_img_size = cv2.resize(nie_rak_img, (50, 50), interpolation=cv2.INTER_LINEAR)
        zdj_bez_raka.append([rak_img_size, 0])

    for img in rak:
        rak_img = cv2.imread(img, cv2.IMREAD_COLOR)
        rak_img_size = cv2.resize(rak_img, (50, 50), interpolation=cv2.INTER_LINEAR)
        zdj_z_rakiem.append([rak_img_size, 1])

    X = []
    y = []

    breast_rak = np.concatenate((zdj_bez_raka, zdj_z_rakiem))
    random.shuffle(images)

    for feature, label in breast_rak:
        X.append(feature)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    print('Ilość zdjęć: {}'.format(len(X)))
    print('Ilość zdjęć bez raka Images: {}'.format(np.sum(y == 0)))
    print('Ilość zdjęć z rakiem Images: {}'.format(np.sum(y == 1)))
    print('Kształt zdjęć: {}'.format(X[0].shape))

    return X, y
