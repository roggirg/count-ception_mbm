import pickle
import scipy
import  numpy as np
from skimage.io import imread
import glob
import os
import matplotlib.pyplot as plt


def getMarkersCells(labelPath, scale, size):
    lab = imread(labelPath)[:, :] / 255

    # print(lab.shape)
    binsize = [scale, scale]
    out = np.zeros(size)
    for i in range(binsize[0]):
        for j in range(binsize[1]):
            out = np.maximum(lab[i::binsize[0], j::binsize[1]], out)

    # print(lab.sum(), out.sum())
    assert np.allclose(lab.sum(), out.sum(), 1)

    return out


def getCellCountCells(markers, xyhw):
    noutputs = 1
    x, y, h, w = xyhw
    types = [0] * noutputs
    for i in range(noutputs):
        types[i] = (markers[y:y + h, x:x + w] == 1).sum()
    return types


def getLabelsCells(markers, img_pad, base_x, base_y, stride, patch_size, framesize_h, framesize_w):
    noutputs = 1
    height = int((img_pad.shape[0]) / stride)
    width = int((img_pad.shape[1]) / stride)
    # print("label size: ", height, width)
    labels = np.zeros((noutputs, height, width))
    # print("base_x", base_x, base_y, framesize_h, height)
    for y in range(0, height):
        for x in range(0, width):
            count = getCellCountCells(markers, (x * stride, y * stride, patch_size, patch_size))
            for i in range(0, noutputs):
                labels[i][y][x] = count[i]

    count_total = getCellCountCells(markers, (0, 0, framesize_h + patch_size, framesize_w + patch_size))
    return labels, count_total


def getTrainingExampleCells(img_raw, framesize_w, framesize_h, labelPath, base_x, base_y,
                            stride, scale, patch_size):
    img = img_raw[base_y:base_y + framesize_h, base_x:base_x + framesize_w]
    img_pad = np.pad(img[:, :, 0], patch_size // 2, "constant")

    markers = getMarkersCells(labelPath, scale, img_raw.shape[0:2])
    markers = markers[base_y:base_y + framesize_h, base_x:base_x + framesize_w]
    markers = np.pad(markers, patch_size, "constant", constant_values=-1)

    labels, count = getLabelsCells(markers, img_pad, base_x, base_y, stride, patch_size, framesize_h, framesize_w)
    return img, labels, count


imgs = []
scale = 2
framesize = int(600/scale)
framesize_h = framesize_w = framesize
patch_size = 32
ef = ((patch_size/1)**2.0)
data_dir = '/PATH_TO/MBM/MBM_data'
datasetfilename = "MBM-dataset.pkl"

for filename in glob.iglob(data_dir + '/*dots*.png'):
    xml = filename.replace("_dots", "")
    imgs.append([xml, filename])


dataset = []
if (os.path.isfile(datasetfilename)):
    print("reading", datasetfilename)
    dataset = pickle.load(open(datasetfilename, "rb"))
else:
    dataset_x = []
    dataset_y = []
    dataset_c = []
    print(len(imgs))
    for path in imgs:
        imgPath = path[0]
        labelPath = path[1]
        print(imgPath)

        im = imread(imgPath)
        img_raw_raw = im  # .mean(axis=(2)) #grayscale

        img_raw = scipy.misc.imresize(img_raw_raw, (int(img_raw_raw.shape[0] / scale), int(img_raw_raw.shape[1] / scale)))
        print(img_raw_raw.shape, " ->>>>", img_raw.shape)

        for base_x in range(0, img_raw.shape[0], framesize_h):
            for base_y in range(0, img_raw.shape[1], framesize_w):

                if (img_raw.shape[1] - base_y < framesize_w) or (img_raw.shape[0] - base_x < framesize_h):
                    print("!!!! Not adding image because size is", img_raw.shape[1] - base_y, img_raw.shape[0] - base_x)
                    continue

                img, lab, count = getTrainingExampleCells(img_raw, framesize_w, framesize_h, labelPath, base_x=0,
                                                          base_y=0, stride=1, scale=scale, patch_size=patch_size)
                print("count ", count)

                if img.shape[0:2] != (framesize_w, framesize_h):
                    print("!!!! Not adding image because size is", img.shape[0:2])

                else:
                    lab_est = [(l.sum() / ef).astype(np.int) for l in lab]

                    assert np.allclose(count, lab_est, 1)

                    dataset.append((img, lab, count))

                    print("lab_est", lab_est, "img shape", img.shape, "label shape", lab.shape)

        print("dataset size", len(dataset))

    print("writing", datasetfilename)
    out = open(datasetfilename, "wb", 0)
    pickle.dump(dataset, out)
    out.close()
print("DONE")

print(len(dataset))

np.random.shuffle(dataset)
np_dataset_x = np.asarray([d[0] for d in dataset])
np_dataset_y = np.asarray([d[1] for d in dataset])
np_dataset_c = np.asarray([d[2] for d in dataset])
np_dataset_x = np_dataset_x.transpose((0, 3, 1, 2))
print("np_dataset_x", np_dataset_x.shape)
print("np_dataset_y", np_dataset_y.shape)
print("np_dataset_c", np_dataset_c.shape)
