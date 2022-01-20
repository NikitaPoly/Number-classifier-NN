import json
import tensorflow as tf
import numpy as np
import timeit
from matplotlib import pyplot as plt

# used to set hyperparameters
hParams = {
    'experimentName': '[128, 10] 20 epochs',
    'datasetProportion': 1.0,
    'valProportion': 0.1,
    'numEpochs': 20,
    'denseLayers': [128,10],
    }
#provided function
def plotPoints(xList, yList, pointLabels=[], xLabel="", yLabel="", title="", filename="pointPlot"):
    plt.figure()
    plt.scatter(xList,yList)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    if pointLabels != []:
        for i, label in enumerate(pointLabels):
            plt.annotate(label, (xList[i], yList[i]))
    filepath = "results/" + filename + ".png"
    plt.savefig(filepath)
    print("Figure saved in", filepath)
#provided function : plots our data
def plotCurves(x, yList, xLabel="", yLabelList=[], title=""):
    fig, ax = plt.subplots()
    y = np.array(yList).transpose()
    ax.plot(x, y)
    ax.set(xlabel=xLabel, title=title)
    plt.legend(yLabelList, loc='best', shadow=True)
    ax.grid()
    filepath = "results/" + title + " " + str(yLabelList) + ".png"
    fig.savefig(filepath)
    print("Figure saved in", filepath)
#given function
def processResults():
    hParamsl, trainResults, testResults = readExperimentalResults(hParams["experimentName"])
    itemsToPlot = ['accuracy', 'val_accuracy']
    plotCurves(x=np.arange(0, hParamsl['numEpochs']),
    yList=[trainResults[item] for item in itemsToPlot],
    xLabel="Epoch",
    yLabelList=itemsToPlot,
    title=hParamsl['experimentName'])
    itemsToPlot = ['loss', 'val_loss']
    plotCurves(x=np.arange(0, hParamsl['numEpochs']),
    yList=[trainResults[item] for item in itemsToPlot],
    xLabel="Epoch",
    yLabelList=itemsToPlot,
    title=hParamsl['experimentName'])
#class for testing
class TEST:
    @staticmethod
    def all():
        TEST.TESToneHiddenLayer()
        TEST.TESTdenseNN()
        TEST.TESTcorrespondingShuffle()
    @staticmethod
    def TESToneHiddenLayer():
        print('\033[93m'+"Testing oneHiddenLayer [128,10]" + "\033[0;0m")
        oneHiddenLayer(get10ClassData(hParams["datasetProportion"],silent=True))
    @staticmethod
    def TESTdenseNN():
        print('\033[93m' + "Testing Multiple layers [128,10]" + "\033[0;0m")
        denseNN(get10ClassData(hParams["datasetProportion"],silent=True))
    @staticmethod
    def TESTcorrespondingShuffle():
        print('\033[93m' + "Testing the correspondingShuffle method" + "\033[0;0m")
        x = [1,2,3,4,5,6,7,8,9]
        y = [1,2,3,4,5,6,7,8,9]
        x,y = correspondingShuffle(x,y)
        for i in range(len(x)):
            print('\033[91m' + "FAIL" + "\033[0;0m") if x[i] != y[i] else print('\033[92m' + "PASS" + "\033[0;0m");
    @staticmethod
    def TESTvalidationProportion():
        print('\033[93m' + "Testing the Validation proportion method" + "\033[0;0m")
        x1,y1,x2,y2,x3,y3 = get10ClassData(1,True)
        print('\033[92m')
        print(len(x1))
        print(len(x2))
        print(len(x3))
        print("\033[0;0m")
    @staticmethod
    def TESTwriteExperimentalResults():
        writeExperimentalResults()
    @staticmethod
    def TESTreadExperimentalResults():
        readExperimentalResults("[128, 10] 20 epochs")
#run Model functions
def oneUnit2Class(dataSubsets):
    x_train, y_train, x_test, y_test = dataSubsets
    firstNNModel = tf.keras.Sequential()
    firstNNModel.add(tf.keras.layers.Dense(1))
    startTime = timeit.default_timer()
    firstNNModel.compile(loss="BinaryCrossentropy",metrics="accuracy",optimizer= "rmsprop")
    elapsedTime = timeit.default_timer() - startTime
    print('\033[94m' + "Construction Time : " + str(elapsedTime) + "\033[0;0m")
    startTime = timeit.default_timer()
    firstNNModel.fit(x_train,y_train,epochs=hParams['numEpochs'])
    elapsedTime = timeit.default_timer() - startTime
    print('\033[94m' + "Training time : " + str(elapsedTime) + "\033[0;0m")
    startTime = timeit.default_timer()
    accuracy = firstNNModel.evaluate(x_test,y_test)
    elapsedTime = timeit.default_timer() - startTime
    print('\033[94m' + "Testing time : " + str(elapsedTime) + " With Accuracy : " + str(accuracy) + "\033[0;0m")
def oneHiddenLayer(dataSubsets):
    x_train, y_train, x_test, y_test = dataSubsets
    secondNNModel = tf.keras.Sequential()
    secondNNModel.add(tf.keras.layers.Dense(128,activation="ReLU"))
    secondNNModel.add(tf.keras.layers.Dense(10))
    startTime = timeit.default_timer()
    secondNNModel.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics="accuracy",optimizer= "rmsprop")
    elapsedTime = timeit.default_timer() - startTime
    print('\033[94m' + "Construction Time : " + str(elapsedTime) + "\033[0;0m")
    startTime = timeit.default_timer()
    secondNNModel.fit(x_train, y_train, epochs=hParams['numEpochs'])
    elapsedTime = timeit.default_timer() - startTime
    print('\033[94m' + "Training time : " + str(elapsedTime) + "\033[0;0m")
    startTime = timeit.default_timer()
    accuracy = secondNNModel.evaluate(x_test, y_test)
    elapsedTime = timeit.default_timer() - startTime
    print('\033[94m' + "Testing time : " + str(elapsedTime) + " With Accuracy : " + str(accuracy) + "\033[0;0m")
    print('\033[92m')
    secondNNModel.summary()
    print(".count_params() = " + str(secondNNModel.count_params()))
    print("\033[0;0m")
def denseNN(dataSubsets):
    x_train, y_train,x_val,y_val, x_test, y_test = dataSubsets
    thirdNNModel = tf.keras.Sequential()
    for i in range(len(hParams["denseLayers"])):
        if i + 1 == len(hParams["denseLayers"]):
            thirdNNModel.add(tf.keras.layers.Dense(hParams["denseLayers"][i]))
        else:
            thirdNNModel.add(tf.keras.layers.Dense(hParams["denseLayers"][i],activation="ReLU"))
    startTime = timeit.default_timer()
    thirdNNModel.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics="accuracy",optimizer="rmsprop")
    elapsedTime = timeit.default_timer() - startTime
    print('\033[94m' + "Construction Time : " + str(elapsedTime) + "\033[0;0m")
    startTime = timeit.default_timer()
    hist = thirdNNModel.fit(x_train, y_train,validation_data=(x_val, y_val) if hParams['valProportion'] != 0.0 else None,epochs=hParams['numEpochs'],verbose=1)
    elapsedTime = timeit.default_timer() - startTime
    print('\033[94m' + "Training time : " + str(elapsedTime) + "\033[0;0m")
    hParams["paramCount"] = thirdNNModel.count_params()
    startTime = timeit.default_timer()
    accuracy = thirdNNModel.evaluate(x_test, y_test)
    elapsedTime = timeit.default_timer() - startTime
    print('\033[94m' + "Testing time : " + str(elapsedTime) + " With Accuracy : " + str(accuracy) + "\033[0;0m")
    print('\033[92m')
    thirdNNModel.summary()
    print(".count_params() = " + str(thirdNNModel.count_params()))
    print("\033[0;0m")
    return hist.history,accuracy
#get Data functions
def get10ClassData(proportion = 1.0,silent=False):
    (x_train, y_train), (x_test, y_test) =  tf.keras.datasets.fashion_mnist.load_data()
    if proportion != 1.0:
        x_train = x_train[:int(len(x_train) * proportion)]
        y_train = y_train[:int(len(y_train) * proportion)]
        x_test = x_test[:int(len(x_test) * proportion)]
        y_test = y_test[:int(len(y_test) * proportion)]
    #set rnage to be 0 - 1
    x_train = x_train / 255
    x_test = x_test / 255
    #reshape from 3d to 1d
    x_train = tf.reshape(x_train,[x_train.shape[0],28*28])
    x_test = tf.reshape(x_test,[x_test.shape[0],28*28])
    #shuffle
    x_train, y_train = correspondingShuffle(x_train, y_train)
    x_test, y_test = correspondingShuffle(x_test, y_test)
    #split data into train, val and test:
    if hParams["valProportion"] > 0.0:
        maxValIndexX = int(x_train.shape[0] * hParams["valProportion"])
        maxValIndexY = int(y_train.shape[0] * hParams["valProportion"])
        x_val = x_train[:maxValIndexX]
        y_val = y_train[:maxValIndexY]
        x_train = x_train[maxValIndexX:]
        y_train = y_train[maxValIndexY:]
        if not silent:
            print(x_train)
            print("With shape : " + str(x_train.shape))
            print(y_train)
            print("With shape : " + str(y_train.shape))
            print(x_test)
            print("With shape : " + str(x_test.shape))
            print(y_test)
            print("With shape : " + str(y_test.shape))
        return x_train, y_train, x_val, y_val, x_test, y_test
    if not silent:
        print(x_train)
        print("With shape : " + str(x_train.shape))
        print(y_train)
        print("With shape : " + str(y_train.shape))
        print(x_test)
        print("With shape : " + str(x_test.shape))
        print(y_test)
        print("With shape : " + str(y_test.shape))
    return x_train, y_train,None, None, x_test, y_test
#shuffle data function
def correspondingShuffle(x,y):
    indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    x = tf.gather(x, shuffled_indices)
    y= tf.gather(y, shuffled_indices)
    return x,y
#write expirement result
def writeExperimentalResults(trainResults,testResults):
    dict = {
        "hParams" : hParams,
        "trainResults" :trainResults,
        "testResults" : testResults
    }
    with open ("results/" + str(hParams["experimentName"])+ ".txt" , "w") as resultFile:
        resultFile.write(json.dumps(dict))
#read data from our files
def readExperimentalResults(filePath):#Finish
    with open("results/" + filePath + ".txt") as file:
        data = json.load(file)
        hParamsl = data["hParams"]
        return hParamsl, data["trainResults"], data["testResults"]
#visualoization functions
def buildValAccuracyPlot():
    itemsToPlot = ['[4, 10] 20 epochs', '[32, 10] 20 epochs','[512, 10] 20 epochs','[256,128,64 10] 20 epochs', '[128, 10] 20 epochs']
    yList = []
    for item in itemsToPlot:
        hParamsl, trainResults, testResults = readExperimentalResults(item)
        yList.append(trainResults["val_accuracy"])
    plotCurves(x=np.arange(0, hParamsl['numEpochs']),
               yList=yList,
               xLabel="Epoch",
               yLabelList=itemsToPlot,
               title="validation accuracies")
def buildTestAccuracyPlot():
    pointLabels = ['[4, 10] 20 epochs', '[32, 10] 20 epochs', '[512, 10] 20 epochs', '[256,128,64 10] 20 epochs','[128, 10] 20 epochs']
    xList = []
    yList = []
    xLabel = "Parameter Count"
    yLabel = "Test Set Accuracy"
    title = "Test Accuracy for Various Models"
    filename = "01HW08"
    for item in pointLabels:
        hParamsl, trainResults, testResults = readExperimentalResults(item)
        xList.append(hParamsl["paramCount"])
        yList.append((testResults[1]))
    plotPoints(xList, yList, pointLabels, xLabel, yLabel, title, filename)
def main2():
    theSeed = 50
    np.random.seed(theSeed)
    tf.random.set_seed(theSeed)
    dataSubsets = get10ClassData(silent=True)
    trainResults, testResults = denseNN(dataSubsets)
    writeExperimentalResults(trainResults, testResults)
    processResults()
def main1():
    # set random seed for testing
    tf.random.set_seed(50)
    np.random.seed(50)
    """
    result = denseNN(get10ClassData(1,True))
    print(result)
    """
    buildValAccuracyPlot()
    buildTestAccuracyPlot()
main1()