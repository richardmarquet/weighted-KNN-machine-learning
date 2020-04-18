import math
#only gets first/second col of each row, would be way to slow otherwise!
def printData(data):
    for d in data:
        print(d)

def getNumFeatures(data):
    return len(data[0])

def getNumSamples(data):
    return len(data)

def assignData(data, missing_data, clean_data):
    for d in data:
        if 1e99 in d:
            missing_data.append(d)
        else:
            clean_data.append(d)

def weightedKNN(data, p, k = 6):
    distance=[]
    for d in data:
        eucl_dist = 0
        for i in range(getNumFeatures(data)):
            if p[i] != 1e99:
                eucl_dist += (d[i] - p[i])**2
            
        eucl_dist = math.sqrt(eucl_dist)
        distance.append((eucl_dist,d))

    distance = sorted(distance)[:k]

    #regular knn
    for i in range(getNumFeatures(data)):
        if p[i] == 1e99:
            p[i] = distance[0][1][i]

    print("new array ->", p)
    #print(distance)

f = open("C://Users//Gaming-Desktop//Desktop//MissingData1.txt", "r")
fl = f.readlines()

print("Loading data. . .")

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

printData(data)
print("Data load complete. . .")
print("Features ->", getNumFeatures(data))
print("Samples ->", getNumSamples(data))

missing_data = []
clean_data = []

#this will seperate the data into two seperate arrays
#one array with no missing data, another with missing data
#we will clasify with the clean data, and then use that to classify missing data!
assignData(data, missing_data, clean_data)

#run the algo
for d in missing_data:
    weightedKNN(clean_data, d, 6)





