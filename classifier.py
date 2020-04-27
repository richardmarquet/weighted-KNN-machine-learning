import math
numClasses = 0
numFeatures = 0
numSamples = 0

def printData(data):
    for d in data:
        print(d)

def median(l, r):
    n = r - l + 1
    n = (n + 1) // 2 - 1
    return n + l

def missing_data_mean_feature(testData):
    for num in range(numFeatures):
        mean = 0
        for d in testData:
            if d[0][num] != 1e99:
                mean += d[0][num]
        mean = mean / numSamples
        for d in testData:
            if d[0][num] == 1e99:
                d[0][num] = mean

def outlierRemover(testData):
    totalVals = []
    for group in testData:
        d = group[0]
        total = 0
        for val in d:
            if val != 1e99:
                total += val
        totalVals.append(total)

    totalVals = sorted(totalVals)
    mid = median(0, len(totalVals))
    q1 = totalVals[median(0, mid)]
    q3 = totalVals[median(mid + 1, len(totalVals))]
    iqr = q3 - q1;

    #anything lower than this
    low_outliers = q1 - 1.5*iqr
    #anything higher than this
    high_outliers = q3 + 1.5*iqr
    n = 0
    for i in range(len(totalVals)):
        if totalVals[i] < low_outliers or totalVals[i] > high_outliers:
            testData.pop(i - n)
            n += 1

    print("Removed", str(n), "outliers")

def weightedKNN(testData, p, k = 6):
    distance=[]
    for group in testData:
        d = group[0]
        eucl_dist = 0
        for i in range(numFeatures):
            if i < len(p) and i < len(d) and p[i] != 1e99 and d[i] != 1e99:
                eucl_dist += (d[i] - p[i])**2
                
        eucl_dist = math.sqrt(eucl_dist)
        distance.append((eucl_dist,group[0],group[1]))
        
    distance = sorted(distance)[:k]
    smallest = distance[0]

    freqList = [0] * numClasses
    
    for d in distance:
        if d[0] != 0:
            freqList[d[2]-1] += 1 / d[0]
        else:
            freqList[d[2]-1] += 9999999

    index = 0
    largestVal = freqList[0]
    for i in range(len(freqList)):
        if freqList[i] > largestVal:
            largestVal = freqList[i]
            index = i
    
    #print(freqList)
    #print("Val chosen ->", index+1)
    return index+1
    
#TrainData
f = open("C://Users//Gaming-Desktop//Desktop//TrainData3.txt", "r")
#TrainLabel
fLabel = open("C://Users//Gaming-Desktop//Desktop//TrainLabel3.txt", "r")
#TestData
fReal = open("C://Users//Gaming-Desktop//Desktop//TestData3.txt", "r")
fl = f.readlines()
f2 = fLabel.readlines()
f3 = fReal.readlines()

print("Loading data. . .")

#loads all labels into 'labels' array!
labels = []
maxNum = 0
for l in f2:
    if l == '\n':
        continue
    if int(l) > maxNum:
        maxNum = int(l)
    labels.append(int(l))
fLabel.close()
numClasses = maxNum

#loads all data into 'data' array!
data = []
num = ""
for line in fl:
    col = []
    for ch in line:
        if ch.isspace() and num:
            col.append(float(num))
            num = ""
        elif not ch.isspace():
            num += ch
    data.append(col)
f.close()

#lodas all test data in 'realData' array
realData = []
num = ""
for line in f3:
    col = []
    for ch in line:
        if (ch == ',' or ch.isspace()) and num:
            if float(num) == 1000000000:
                print("yup")
                col.append(float(1e99))
            else:
                col.append(float(num))
            num = ""
        elif not ch.isspace() or ch != ',':
            num += ch
    realData.append(col)
fReal.close()

numFeatures = len(data[0])
numSamples = len(data)

#printData(data)
print("Data load complete. . .")
print("Features ->", numFeatures)
print("Samples ->", numSamples)
print("Creating test data. . .")

#The classification data
testData = []
for i in range(numSamples):
    testData.append((data[i],labels[i]))

print("Removing outliers. . .")
outlierRemover(testData)

missing_data_mean_feature(testData)

#total = 0
#for i in range(len(testData[:20])):
#    guess = weightedKNN(testData[20:], testData[i][0], 6)
#    print("Actual value ->", testData[i][1])
#    if guess == testData[i][1]:
#        total += 1
#print(total / 20)

print("Starting KNN. . .")
sampleNum = 0

#Output File
result = open("C://Users//Gaming-Desktop//Desktop//Machine Learning Project//MarquetClassification3.txt", "w")
for i in range(len(realData)):
    val = weightedKNN(testData, realData[i], 6)
    #print("Sample", sampleNum, "was predicted to be a value of ->", val)
    sampleNum += 1
    result.write(str(val) + "\n")
result.close()

    


