import os
import numpy as np
import cv2


class DataProvider():
    "this class creates machine-written text for a word list. TODO: change getNext() to return your samples."

    def __init__(self, data_images):
        self.data_images = data_images
        self.idx = 0

    def hasNext(self):
        "are there still samples to process?"
        return self.idx < len(self.data_images)

    def getNext(self):
        "TODO: return a sample from your data as a tuple containing the text and the image"
        # img = np.ones((32, 128), np.uint8)*255
        img = cv2.imread(self.data_images[self.idx])
        word = self.data_images[self.idx].split('/')[-1].split('_')[0]
        #word = self.data_images[self.idx].split('/')[-1].split('_')[1].split('.')[0]
        print('data generated : ', self.data_images[self.idx])
        self.idx += 1
        
        return (word, img)


def createIAMCompatibleDataset(dataProvider):
    "this function converts the passed dataset to an IAM compatible dataset"

    # create files and directories
    
    if not os.path.exists('dataset/data'):
        os.makedirs('dataset/data')
    f = open('dataset/data/words.txt', 'w+')
    if not os.path.exists('dataset/data/img'):
        os.makedirs('dataset/data/img')

    # go through data and convert it to IAM format
    ctr = 0
    while dataProvider.hasNext():
        sample = dataProvider.getNext()

        # write img
        cv2.imwrite('dataset/data/img/image-%d.png' % ctr, sample[1])

        # write filename, dummy-values and text
        line = 'img/image-%d.png' % ctr + '\t' + sample[0] + '\n'
        f.write(line)

        ctr += 1


if __name__ == '__main__':
    # words = ['some', 'words', 'for', 'which', 'we', 'create', 'text-images']
    dataset_path = 'data/'
    data_images = []

    for img in os.listdir(dataset_path):
        data_images.append(os.path.join(dataset_path, img))

    dataProvider = DataProvider(data_images)
    # print('1')
    createIAMCompatibleDataset(dataProvider)
    # print('2')
