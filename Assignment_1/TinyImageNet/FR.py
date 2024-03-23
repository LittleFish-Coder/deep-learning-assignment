#!/usr/bin/env python
# coding: utf-8


import os
import numpy as np
from numpy import linalg as LA
from FeatureExtraction import HOG, ColorHistogram, SIFT
import cv2


def prepare_data():
    # train test split percentage -> 80% train, 20% test
    train_test_split = 0.8

    dd = os.listdir("TIN")
    f1 = open("train.txt", "w")
    f2 = open("test.txt", "w")
    print(f"start data spliting, train_test_split = {train_test_split}")
    for i in range(len(dd)):
        d2 = os.listdir("TIN/%s/images/" % (dd[i]))
        # training data
        for j in range(int(len(d2) * train_test_split)):  # 80% train
            str1 = "TIN/%s/images/%s" % (dd[i], d2[j])
            f1.write("%s %d\n" % (str1, i))  # write to train.txt
        # testing data
        for j in range(int(len(d2) * train_test_split), len(d2)):
            str1 = "TIN/%s/images/%s" % (dd[i], d2[-1])
            f2.write("%s %d\n" % (str1, i))  # write to test.txt
        # end of training and testing data for class i

    f1.close()
    f2.close()

    print("data spliting done")


# prepare_data()  # data preparation (1 time only)


# downsample image to 32x32 for faster processing
def resize_img(img, size=(32, 32)):
    img = cv2.imread(img)
    img = cv2.resize(img, size)
    return img


def load_img(f):
    f = open(f)
    lines = f.readlines()
    imgs, lab = [], []
    for i in range(len(lines)):
        fn, label = lines[i].split(" ")
        im1 = resize_img(fn)

        """===============================
        影像處理的技巧可以放這邊，來增強影像的品質
        
        ==============================="""

        # feature extraction from FeatureExtraction.py
        # im1 = HOG(im1)
        # im1 = ColorHistogram(im1)
        # keypoints, im1 = SIFT(im1)
        vec = np.reshape(im1, [-1])
        # print(vec.shape)
        imgs.append(vec)
        lab.append(int(label))

    imgs = np.asarray(imgs, np.float32)
    lab = np.asarray(lab, np.int32)
    return imgs, lab


x, y = load_img("train.txt")
tx, ty = load_img("test.txt")


# ======================================
# X就是資料，Y是Label，請設計不同分類器來得到最高效能
# 必須要計算出分類的正確率
# ======================================


def KNN(x, y):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    print("start KNN")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(x, y)
    print("KNN training done")
    return model


def SVM(x, y):
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score

    print("start SVM")
    model = SVC(kernel="linear")
    model.fit(x, y)
    print("SVM training done")
    return model


def CNN_Method():
    return None


# main
if __name__ == "__main__":

    # data preparation
    # prepare_data()

    # data transformation (feature extraction)
    # x, y = load_img("train.txt")
    # tx, ty = load_img("test.txt")

    print("start training")
    # training
    # model = train(x, y)

    # testing
    # test(model, tx, ty)
    print("done")
