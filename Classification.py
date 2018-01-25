import random
import sys
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from mnist_extractor import Extractor

trainImages, trainLabels, testImages, testLabels = Extractor.getData()

# classifiers below are only some of all classifiers available in scikit learn
# I chose these based on an example comparing different classifiers available
# in scikit-learn.org

# I used below file to log the result for the whole dataset. afterwards, the
# number of training images is set to 2000 so the code would run faster. the
# result of training with 60000 images is available in log.md. don't uncomment
# the related lines below, for the log will be replaced.
# log = open('log.md', 'w')
# sys.stdout = log

# 1. Random Forest
print('*Random Forest*')
rfc = RandomForestClassifier(n_estimators=50, n_jobs=10, verbose=0)
rfc.fit(trainImages, trainLabels)
print('  accuracy:', rfc.score(testImages, testLabels), end='\n\n')

# 2. Gradient Booster
print('*Gradient Booster*')
gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=0.15, verbose=0)
gbc.fit(trainImages, trainLabels)
print('  Accuracy:', gbc.score(testImages, testLabels), end='\n\n')

# 3. K-Neighbours
print('*K-Neighbours*')
knc = KNeighborsClassifier(5, algorithm='auto')
knc.fit(trainImages, trainLabels)
print('  Accuracy:', knc.score(testImages, testLabels), end='\n\n')

# 4. MLP
print('*Multi-Layer Perceptron')
mlpc = MLPClassifier()
mlpc.fit(trainImages, trainLabels)
print('  Accuracy:', mlpc.score(testImages, testLabels), end='\n\n')

# 5. SVC
print('*Support Vector*')
svc = SVC(gamma=0.002)
svc.fit(trainImages, trainLabels)
print('  Accuracy:', svc.score(testImages, testLabels), end='\n\n')

# 6. Ada boost
print('*Ada Boost*')
abc = AdaBoostClassifier()
abc.fit(trainImages, trainLabels)
print('  Accuracy:', abc.score(testImages, testLabels), end='\n\n')

# 7. Decision Tree
print('*Decision Tree*')
dtc = DecisionTreeClassifier()
dtc.fit(trainImages, trainLabels)
print('  Accuracy:', dtc.score(testImages, testLabels), end='\n\n')

# 8. Quadratic Discriminant Analysis
print('*Quadratic Discriminant Analysis*')
qdac = QuadraticDiscriminantAnalysis()
qdac.fit(trainImages, trainLabels)
print('  Accuracy:', qdac.score(testImages, testLabels), end='\n\n')

# 9. Gaussian Process
print('*Gaussian Process*')
gpc = GaussianProcessClassifier()
gpc.fit(trainImages, trainLabels)
print('  Accuracy:', gpc.score(testImages, testLabels), end='\n\n')

# 10. Gaussian Naive Bayes
print('*Gaussian Naive Bayes*')
gnbc = GaussianNB()
gnbc.fit(trainImages, trainLabels)
print('  Accuracy:', gnbc.score(testImages, testLabels), end='\n\n')

# two examples
print('- Random examples')
for cnt in range(2):
    i = random.randint(0, len(testImages))
    for idx in range(len(testImages[i])):
        p = testImages[i][idx]
        if p == 0:
            print('__', end='')
        else:
            print('##', end='')
        if idx % 28 == 27:
            print()

    print('\nexpected output:', testLabels[i])
    print('RF predicted output:', rfc.predict([testImages[i]])[0])
    print('GB predicted output:', gbc.predict([testImages[i]])[0])
    print('KN predicted output:', knc.predict([testImages[i]])[0])
    print('MLP predicted output:', mlpc.predict([testImages[i]])[0])
    print('SV predicted output:', svc.predict([testImages[i]])[0])
    print('AdaB predicted output:', abc.predict([testImages[i]])[0])
    print('DT predicted output:', dtc.predict([testImages[i]])[0])
    print('QDA predicted output:', qdac.predict([testImages[i]])[0])
    print('GP predicted output:', gpc.predict([testImages[i]])[0])
    print('GNB predicted output:', gnbc.predict([testImages[i]])[0])
