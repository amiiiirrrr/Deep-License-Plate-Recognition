""" a modified version of CRNN torch repository https://github.com/bgshih/crnn/blob/master/tool/create_dataset.py """

# import fire
import os
import shutil
import lmdb
import cv2
from modules.IMA import DataProvider, createIAMCompatibleDataset
import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(inputPath, gtFile, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        inputPath  : input folder path where starts imagePath
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    with open(gtFile, 'r', encoding='utf-8') as data:
        datalist = data.readlines()

    nSamples = len(datalist)
    for i in range(nSamples):
        imagePath, label = datalist[i].strip('\n').split('\t')
        imagePath = os.path.join(inputPath, imagePath)

        # # only use alphanumeric data
        # if re.search('[^a-zA-Z0-9]', label):
        #     continue

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def create_dataset():
    """ Create train lmdb dataset"""
    os.makedirs(f'./dataset/train', exist_ok=True)
    os.makedirs(f'./dataset/valid', exist_ok=True)
    up_paths = os.listdir('./dataset/up')
    print(len(up_paths))
    valid_percent = 0.1 
    valid_paths = up_paths[0:int(valid_percent * len(up_paths))]
    train_paths = up_paths[int(valid_percent * len(up_paths)):]
    for valid_paht in valid_paths:
        # print(valid_paht)
        shutil.copyfile(f'./dataset/up/{valid_paht}', './dataset/valid/'+ valid_paht)
    for train_paht in train_paths:
        shutil.copyfile(f'./dataset/up/{train_paht}', './dataset/train/'+ train_paht)

    dataset_path = 'dataset/train/'
    data_images = []

    for img in os.listdir(dataset_path):
        data_images.append(os.path.join(dataset_path, img))

    dataProvider = DataProvider(data_images)
    createIAMCompatibleDataset(dataProvider)
    createDataset(inputPath = 'dataset/data/', gtFile = 'dataset/data/words.txt', outputPath = 'dataset/train_lmdb')
    shutil.rmtree('dataset/data')
    """ Create validation lmdb dataset"""
    dataset_path = 'dataset/valid/'
    data_images = []

    for img in os.listdir(dataset_path):
        data_images.append(os.path.join(dataset_path, img))

    dataProvider = DataProvider(data_images)
    createIAMCompatibleDataset(dataProvider)
    createDataset(inputPath = 'dataset/data/', gtFile = 'dataset/data/words.txt', outputPath = 'dataset/valid_lmdb')
    shutil.rmtree('dataset/data/')
    # fire.Fire(createDataset)
