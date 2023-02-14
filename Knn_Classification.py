#!/usr/bin/env python3

import cv2
import sys
import csv
import time
import numpy as np

### Load training images and labels

imageDirectory = './imgs/train_images/'

with open(imageDirectory + 'train.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)

# this line reads in all images listed in the file, and resizes them to 33x25 pixels
#train = np.array( [np.array(cv2.resize(cv2.imread(imageDirectory +lines[i][0]+".png",0),(33,25))) for i in range(len(lines))])

train = np.array( [np.array(cv2.imread(imageDirectory +lines[i][0]+".jpg",3)) for i in range(len(lines))])
# here we reshape each image into a long vector and ensure the data type is a float (which is what KNN wants)
#train_data = train.flatten().reshape(len(lines), 33*25*3)
#train_data = train_data.astype(np.float32)
train_crop = []
def crop_orange(original_img, visualize=False):
    #cv2.imshow("original", original_img);cv2.waitKey();cv2.destroyAllWindows()
    
    hsv = cv2.cvtColor(original_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,(0, 120, 20), (180, 255, 255))
    pos = np.where(mask>0)
    x_mean = np.mean(pos[0])
    y_mean = np.mean(pos[1])
    h, w, _ = original_img.shape
    
    if (abs(x_mean) < 0.2*w) or (abs(x_mean) > 0.8*w) or (abs(y_mean) < 0.2*h) or (abs(y_mean) > 0.8*h):
        mask = np.zeros(mask.shape)

    if visualize:
        print("show mask")
        cv2.imshow("mask", mask);cv2.waitKey();cv2.destroyAllWindows()
    
    mask[pos[0][np.where(abs(pos[0] - x_mean) > w / 3.0)], pos[1][np.where(abs(pos[0] - x_mean) > w / 3.0)]] = 0
    mask[pos[0][np.where(abs(pos[1] - y_mean) > h / 3.0)], pos[1][np.where(abs(pos[1] - y_mean) > h / 3.0)]] = 0
    pos = np.where(mask>0)
    
    if visualize:
        print("show mask")
        cv2.imshow("mask", mask);cv2.waitKey();cv2.destroyAllWindows()
    
    if x_mean < w * 0.8 and x_mean > w * 0.2:
        mask[pos[0][np.where(pos[0]<0.1*w)], :] = 0
        mask[pos[0][np.where(pos[0]>0.9*w)], :] = 0
    
    if y_mean < h * 0.8 and y_mean > h * 0.2:
        mask[:, pos[1][np.where(pos[1]<0.1*h)]] = 0
        mask[:, pos[1][np.where(pos[1]>0.9*h)]] = 0
    
    if visualize:
        print("show mask")
        cv2.imshow("mask", mask);cv2.waitKey();cv2.destroyAllWindows()
    
    n_masked_pts = np.sum(mask)
    n_pts = w*h
    pos_x, pos_y = np.where(mask>0)[0], np.where(mask>0)[1]
    if len(pos_x) > 0:
        if (abs(np.min(np.where(mask>0)[0]) - np.max(np.where(mask>0)[0])) < w/12) or (abs(np.min(np.where(mask>0)[1]) - np.max(np.where(mask>0)[1])) < h/12):
            mask = np.zeros(mask.shape)
    
    if visualize:
        print("show mask")
        cv2.imshow("mask", mask);cv2.waitKey();cv2.destroyAllWindows()
    
    # crop image
    if np.sum(mask) == 0:
        img_crop = original_img[int(w/3):int(w*2/3), int(h/3):int(h*2/3), :]
    else:
        x_min, x_max = np.min(np.where(mask>0)[0]), np.max(np.where(mask>0)[0])
        y_min, y_max = np.min(np.where(mask>0)[1]), np.max(np.where(mask>0)[1])
        img_crop = original_img[max(0, int(0.9*x_min)): min(w, int(1.1* x_max)), max(0, int(0.9*y_min)): min(h, int(1.1*y_max)), :]

    #cv2.imshow("croped image", img_crop);cv2.waitKey();cv2.destroyAllWindows()
    
    return img_crop

for i, original_img in enumerate(train):
    print(i)
    ## find orange rectagnular
    #cv2.imshow("original", original_img);cv2.waitKey();cv2.destroyAllWindows()
    img_crop = crop_orange(original_img)
    train_crop.append(cv2.resize(img_crop, (33,25)))
    #cv2.imshow("image with corner", original_img);cv2.waitKey();cv2.destroyAllWindows()
    #cv2.waitKey(0)
    #cv2.imshow("cropped image", img_crop);cv2.waitKey();cv2.destroyAllWindows()
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
train_crop_array = np.array(train_crop)

train_data = train_crop_array.flatten().reshape(len(lines), 33*25*3)
train_data = train_data.astype(np.float32)

# read in training labels
train_labels = np.array([np.int32(lines[i][1]) for i in range(len(lines))])

### Train classifier
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

if(__debug__):
	Title_images = 'Original Image'
	Title_resized = 'Image Resized'
	cv2.namedWindow( Title_images, cv2.WINDOW_AUTOSIZE )

### Run test images
imageDirectory_test = './imgs/test_images/'
with open(imageDirectory_test + 'test.txt', 'r') as f:
    reader = csv.reader(f)
    lines = list(reader)
print(lines)
correct = 0.0
confusion_matrix = np.zeros((6,6))

for i in range(len(lines)):
#for i in range(147, 149):
    test_img = np.array(cv2.imread(imageDirectory_test+lines[i][0]+".jpg",3))
    croped_img = crop_orange(test_img)
    test_img = cv2.resize(croped_img, (33,25))
    
    if len(sys.argv) == 1:
        pass
    else:
        if str(sys.argv[1] == '-v'):
            cv2.imshow(Title_images, original_img)
            cv2.imshow(Title_resized, test_img)
            k = cv2.waitKey()
            if k==27:    # Esc key to stop
                break
    test_img = test_img.flatten().reshape(1, 33*25*3)
    test_img = test_img.astype(np.float32)

    test_label = np.int32(lines[i][1])

    ret, results, neighbours, dist = knn.findNearest(test_img, 2)

    if test_label == ret:
        print(str(lines[i][0]) + " Correct, " + str(ret))
        correct += 1
        confusion_matrix[np.int32(ret)][np.int32(ret)] += 1
    else:
        confusion_matrix[test_label][np.int32(ret)] += 1
        
        print(str(lines[i][0]) + " Wrong, " + str(test_label) + " classified as " + str(ret))
        print("\tneighbours: " + str(neighbours))
        print("\tdistances: " + str(dist))



print("\n\nTotal accuracy: " + str(correct/len(lines)))
print(confusion_matrix)
