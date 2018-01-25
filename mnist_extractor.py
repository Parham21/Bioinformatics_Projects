from struct import *
from sklearn.utils.testing import all_estimators


class Extractor:
    @staticmethod
    def getData():
        trainImages = open('dataset/train-images.idx3-ubyte', 'rb')
        header = trainImages.read(16)
        magicNum, size, rows, cols, = Struct('>IIII').unpack(header)
        # print(magicNum, size, rows, cols)
        imagesMat = []
        print('extracting train images... ', end='', flush=True)
        for cnt in range(2000):
            image = []
            for r in range(rows):
                for c in range(cols):
                    p, = Struct('>B').unpack(trainImages.read(1))
                    image.append(p)
            imagesMat.append(image)
        print('Done')
        # test: reading images correctly
        # for i in imagesMat:
        #     for row in i:
        #         for e in row:
        #             if e == 0:
        #                 print('__', end='')
        #             else:
        #                 print('##', end='')
        #         print()

        trainLabels = open('dataset/train-labels.idx1-ubyte', 'rb')
        header = trainLabels.read(8)
        magicNum, size, = Struct('>II').unpack(header)
        # print(magicNum, size)
        labelsMat = []
        print('extracting train labels... ', end='', flush=True)
        for cnt in range(2000):
            labelsMat.append(Struct('>B').unpack(trainLabels.read(1))[0])
        print('Done')

        # test: reading labels correctly
        # for l in labelsMat:
        #     print(l)

        testImages = open('dataset/t10k-images.idx3-ubyte', 'rb')
        header = testImages.read(16)
        magicNum, size, rows, cols, = Struct('>IIII').unpack(header)
        # print(magicNum, size, rows, cols)
        testImagesMat = []
        print('extracting  test images... ', end='', flush=True)
        for cnt in range(100):
            image = []
            for r in range(rows):
                for c in range(cols):
                    p, = Struct('>B').unpack(testImages.read(1))
                    image.append(p)
            testImagesMat.append(image)
        print('Done')

        testLabels = open('dataset/t10k-labels.idx1-ubyte', 'rb')
        header = testLabels.read(8)
        magicNum, size, = Struct('>II').unpack(header)
        # print(magicNum, size)
        testLabelsMat = []
        print('extracting  test labels... ', end='', flush=True)
        for cnt in range(100):
            testLabelsMat.append(Struct('>B').unpack(testLabels.read(1))[0])
        print('Done\n')
        return imagesMat, labelsMat, testImagesMat, testLabelsMat


# to get a list of all classifiers
# estimators = all_estimators()
# ae = open('allClassifiers.md', 'w')
# for name, class_ in estimators:
#     if hasattr(class_, 'predict'):
#         ae.write(name)
#         ae.write('\n')
