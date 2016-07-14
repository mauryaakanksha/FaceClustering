from __future__ import division
import numpy as np
import cPickle as pkl
from stacked_autoencoder import SdA
import theano
import cv2
import io
import math

from picamera.array import PiRGBArray
from picamera import PiCamera
import time

N_DIM = 100
PARTITION = 10
IS_KMEANS = 1
train_mean = np.load('new_data/store_mean.npy')

def label_faces_from_video(centers):

    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    camera.sharpness = 50
    rawCapture = PiRGBArray(camera, size=(640, 480))
     
    face_cascade = cv2.CascadeClassifier('/home/pi/mainak/opencv-3.0.0/data/haarcascades/haarcascade_frontalface_default.xml')
    
    # allow the camera to warmup
    time.sleep(0.1)

    # loading the trained model
    model_file = file('models/pretrained_model.save', 'rb')
    sda = pkl.load(model_file)
    model_file.close()
    
    get_single_encoded_data = sda.single_encoder_function()

    t = 1
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image, face_images = capture_and_detect(frame, face_cascade)
        
        for face in face_images:
            encoded_x = get_single_encoded_data(train_x=face)
            if (IS_KMEANS == 1):
                label_x, dist = get_kmeans_labels(centers, encoded_x)
            else:    
                label_x = cluster.get_tseries_labels(encoded_x,t)
            print("This is person: ", label_x, dist)
            t += 1
            

    	# show the frame
    	cv2.imshow("Frame", image)
    	key = cv2.waitKey(1) & 0xFF
     
    	# clear the stream in preparation for the next frame
    	rawCapture.truncate(0)
     
    	# if the `q` key was pressed, break from the loop
    	if key == ord("q"):
    		break

def get_kmeans_labels(centers, x):
    dist = []
    for center in centers:
        dist.append(np.linalg.norm(center-x))
    return np.argmin(np.asarray(dist)), min(np.asarray(dist))

def capture_and_detect(frame, face_cascade):
    image = frame.array
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im_gray, 1.3, 5)
    face_images = []
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        face_gray = np.array(im_gray[y:y+h, x:x+w], 'uint8')
        face_sized = cv2.resize(face_gray, (30, 30))

        flat_face = face_sized.reshape(1, face_sized.shape[0]*face_sized.shape[1])
        flat_face = flat_face/255
        face_x = flat_face - train_mean

        face_images.append(face_x)
    return image, face_images

def cluster_train_data(cluster):
    
    train_set = np.load('new_data/train_faces.npy')
    test_set = np.load('new_data/test_faces.npy')

    tr_x = [i[0] for i in train_set]
    tr_y = [i[1] for i in train_set]
    te_x = [i[0] for i in test_set]
    te_y = [i[1] for i in test_set]
    
    train_set_x = theano.shared(value=np.asarray(tr_x), borrow=True)
    test_set_x  = theano.shared(value=np.asarray(te_x), borrow=True)
    
    train_set_l = np.asarray(tr_y)
    test_set_l  = np.asarray(te_y)
    
    # compute number of minibatches for training, validation and testing
    n_train_data = train_set_x.get_value(borrow=True).shape[0]
    print "n_train_data: ", n_train_data
    n_test_data = test_set_x.get_value(borrow=True).shape[0]
    print "n_test_data: ", n_test_data

    train_x = np.zeros((n_train_data, N_DIM), dtype=np.float32)
    test_x  = np.zeros((n_test_data, N_DIM), dtype=np.float32)
    
    # loading the trained model
    model_file = file('models/pretrained_model.save', 'rb')
    sda = pkl.load(model_file)
    model_file.close()
    
    get_encoded_data = sda.encoder_function(train_set_x=train_set_x)
    get_single_encoded_data = sda.single_encoder_function()

    first = 1
    for i in range(n_train_data):
        encoded_x = get_encoded_data(index=i)
        if (IS_KMEANS == 1):
            train_x[i] = encoded_x
        else:
            cluster.get_dimension_info(encoded_x, first)
            first =0
    
    center = []
    if (IS_KMEANS == 1):
        #flags = cv2.KMEANS_RANDOM_CENTERS
        flags = cv2.KMEANS_PP_CENTERS
        # Apply KMeans
        compactness, labels, centers = cv2.kmeans(data=train_x, K=3, bestLabels=None, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100000, 0.001), attempts=10, flags=flags)
        #print "Error: ", compactness, len(labels)
        get_accuracy(train_x, train_set_l, labels)

        test_labels = []
        for i in range(n_test_data):
            encoded_x = get_single_encoded_data(train_x=test_set_x.get_value(borrow=True)[i:i+1])
            label, _ = get_kmeans_labels(centers, encoded_x)
            test_labels.append(label)
            test_x[i] = encoded_x
        get_accuracy(test_x, test_set_l, test_labels)
        return centers
    else:
        t = 1
        for data in train_x:
            cluster.get_tseries_labels(data,t)
            t += 1
    #return centers

def get_accuracy(data_x, data_y, labels):
    A = []
    B = []
    C = []
    A_l1 =0
    A_l2 =0
    A_l3 =0
    B_l1 =0
    B_l2 =0
    B_l3 =0
    C_l1 =0
    C_l2 =0
    C_l3 =0
    # Now split the data depending on their labels
    for i in xrange(len(labels)):
        if (labels[i] == 0):
            if data_y[i] == "label1":
                A_l1 += 1
            elif data_y[i] == "label2":
                A_l2 += 1
            elif data_y[i] == "label3":
                A_l3 += 1
            A.append(data_x[i])
        elif (labels[i] == 1):
            if data_y[i] == "label1":
                B_l1 += 1
            elif data_y[i] == "label2":
                B_l2 += 1
            elif data_y[i] == "label3":
                B_l3 += 1
            B.append(data_x[i])
        elif (labels[i] == 2):
            if data_y[i] == "label1":
                C_l1 += 1
            elif data_y[i] == "label2":
                C_l2 += 1
            elif data_y[i] == "label3":
                C_l3 += 1
            C.append(data_x[i])
    print "Length: ", len(A), len(B), len(C)
    len_A = len(A)
    len_B = len(B)
    len_C = len(C)
    
    max_A = max(A_l1,A_l2,A_l3)
    max_B = max(B_l1,B_l2,B_l3)
    max_C = max(C_l1,C_l2,C_l3)

    #accuracy = (max_A/len_A) + (max_B/len_B) + (max_C/len_C)
    accuracy = (max_A + max_B + max_C)/len(labels)
    print "Acc: ", accuracy
    
    print "Cluster A Count: ", A_l1, A_l2, A_l3
    print "Cluster B Count: ", B_l1, B_l2, B_l3
    print "Cluster C Count: ", C_l1, C_l2, C_l3


class Coordinates(object):

    def __init__(self, c):
        coord  = []
        for i in c:
            coord.append(i)
            i+=1
        self.coords = coord

    def __hash__(self):
        return hash(str(self.coords))
        
    def __eq__(self, other):
        other_list = (other).coords
        if len(other_list) != len(self.coords):
            return False
        i = 0
        while i < len(self.coords):
            if int(self.coords[i]) != int(other_list[i]):
                return False
            i += 1
        return True

    def getCoords(self):
        coord = []
        for c in self.coords:
            coord.append(c)
        return coord

    def getDimension(self, d):
        if self.coords != None and len(self.coords) > d:
            return self.coords[d]
        else:
           # print "Cannot get selected value"
            return None

    def setDimension(self, d, val):
        if self.coords != None and len(self.coords) > d:
            self.coords[d] = val
        else:
            print "Cannot set selected value"

    def getSize(self):
        return len(self.coords)

    def equals(self, other):

        other_list = (other).coords
        if len(other_list) != len(self.coords):
            return False
        i = 0
        while i < len(self.coords):
            if int(self.coords[i]) != int(other_list[i]):
                return False
            i += 1
        return True

    def __str__(self):
        return self.coords.__str__()
# ============================================================================================================================================================================

class ATTRIBUTE:
    DENSE = u'DENSE'
    TRANSITIONAL = u'TRANSITIONAL'
    SPARSE = u'SPARSE'

#  Characteristic vector of a grid
class Grid(object):

    visited = False
    last_time_updated = 0
    last_time_element_added = 0
    grid_density = 0.0
    grid_attribute = ATTRIBUTE.SPARSE
    attraction_list = list()
    cluster = -1
    attributeChanged = False
    DIMENSION =0
    DIMENSION_UPPER_RANGE =0 
    DIMENSION_LOWER_RANGE =0 
    DIMENSION_PARTITION = 0
    TOTAL_GRIDS =0
    decay_factor =0 
    dense_threshold =0 
    sparse_threshold =0
    correlation_threshold =0
    

    def __init__(self, v, c, tg, D, attr, dim, dim_upper, dim_lower, dim_par, total_grids, decay, d_thres, s_thres,c_thres):
        self.visited = v
        self.cluster = c
        self.last_time_updated = tg
        self.grid_density = D
        self.grid_attribute = attr
        self.DIMENSION = dim
        self.DIMENSION_UPPER_RANGE = dim_upper 
        self.DIMENSION_LOWER_RANGE =dim_lower 
        self.DIMENSION_PARTITION = dim_par
        self.TOTAL_GRIDS = total_grids
        self.decay_factor = decay
        self.dense_threshold = d_thres
        self.sparse_threshold = s_thres
        self.correlation_threshold = c_thres
        self.attraction_list = list()


    def __hash__(self):
        return hash(str(self.name))
    
    def __eq__(self, other):
        return str(self.name) == str(other.name)


    def setVisited(self, v):
        self.visited = v

    def isVisited(self):
        return self.visited

    def setCluster(self, c):
        self.cluster = c

    def getCluster(self):
        return self.cluster

    def setLastTimeUpdated(self, tg):
        self.last_time_updated = tg

    def getLastTimeUpdated(self):
        return self.last_time_updated

    def setLastTimeElementAdded(self, tg):
        self.last_time_element_added = tg

    def getLastTimeElementAdded(self):
        return self.last_time_element_added

    def getGridDensity(self):
        return self.grid_density

    def updateGridDensity(self, time):
        self.grid_density = self.grid_density * (math.pow(self.decay_factor, time - self.last_time_updated)) + 1

    def updateDecayedGridDensity(self, time):
        self.grid_density = self.grid_density * (math.pow(self.decay_factor, time - self.last_time_updated))

    def isAttributeChangedFromLastAdjust(self):
        return self.attributeChanged

    def setAttributeChanged(self, val):
        self.attributeChanged = val

    def isDense(self):
        return self.grid_attribute == ATTRIBUTE.DENSE

    def isTransitional(self):
        return self.grid_attribute == ATTRIBUTE.TRANSITIONAL

    def isSparse(self):
        return self.grid_attribute == ATTRIBUTE.SPARSE

    def getGridAttribute(self):
        str_ = ""
        if self.isDense():
            str_ = "DENSE"
        if self.isTransitional():
            str_ = "TRANSITIONAL"
        if self.isSparse():
            str_ = "SPARSE"
        return str_

    def updateGridAttribute(self):
        avg_density = 1.0 / (self.TOTAL_GRIDS * (1 - self.decay_factor))
        if self.grid_attribute != ATTRIBUTE.DENSE and self.grid_density >= self.dense_threshold * avg_density:
            self.attributeChanged = True
            self.grid_attribute = ATTRIBUTE.DENSE
        elif self.grid_attribute != ATTRIBUTE.SPARSE and self.grid_density <= self.sparse_threshold * avg_density:
            self.attributeChanged = True
            self.grid_attribute = ATTRIBUTE.SPARSE
        elif self.grid_attribute != ATTRIBUTE.TRANSITIONAL and self.grid_density > self.sparse_threshold * avg_density and self.grid_density < self.dense_threshold * avg_density:
            self.attributeChanged = True
            self.grid_attribute = ATTRIBUTE.TRANSITIONAL

    def setInitialAttraction(self, attrL):
        for i in attrL:
            self.attraction_list.append(i)


    def normalizeAttraction(self, attr_list):

        total_attr = 0.0
        i = 0
        while i < 2 * self.DIMENSION + 1:
            total_attr += attr_list[i]
            i += 1
        if total_attr <= 0:
            return
        attr = float()
        #  normalize
        i = 0
        while i < 2 * self.DIMENSION + 1:
            attr = attr_list[i]
            attr_list[i]= attr / total_attr
            i += 1


    def getAttraction(self, data_coords, grid_coords):
        attr_list = list()
        i = 0

        while i < 2 * self.DIMENSION + 1:
            attr_list.append(1.0)
            i += 1
        last_element = 2 * self.DIMENSION
        i = 0
        closeToBigNeighbour = False
        while i < len(grid_coords):
            upper_range = self.DIMENSION_UPPER_RANGE[i]
            lower_range = self.DIMENSION_LOWER_RANGE[i]
            num_of_partitions = self.DIMENSION_PARTITION
            partition_width = (upper_range - lower_range) / (num_of_partitions);
            center = grid_coords[i]*partition_width + partition_width/2.0;
            radius = partition_width / 2.0
            epsilon = 0.6*radius
            if data_coords[i] > center:
                closeToBigNeighbour = True
            if (radius - epsilon) > abs(data_coords[i] - center):
                attr_list[2 * i] = 0.0
                attr_list[2 * i + 1] = 0.0
                attr_list[last_element] = 1.0
            else:
                if closeToBigNeighbour:
                    weight1 = ((epsilon - radius) + (data_coords[i] - center))
                    weight2 = ((epsilon + radius) - (data_coords[i] - center))
                    prev_attr = attr_list[2 * i]
                    attr_list[2 * i] = prev_attr * weight1
                    attr_list[2 * i + 1] = 0.0
                    k =0
                    while k < 2 * self.DIMENSION + 1:
                        if k != 2 * i and k != 2 * i + 1:
                            prev_attr = attr_list[k]
                            attr_list[k] = prev_attr * weight2
                        k = k + 1
                else:
                    weight1 = ((epsilon - radius) - (data_coords[i] - center))
                    weight2 = ((epsilon + radius) + (data_coords[i] - center))
                    prev_attr = attr_list[2 * i + 1]
                    attr_list[2 * i + 1] = prev_attr * weight1
                    attr_list[2 * i] =  0.0
                    k =0
                    while k < 2 * self.DIMENSION + 1:
                        if k != 2 * i and k != 2 * i + 1:
                            prev_attr = attr_list[k]
                            attr_list[k] = prev_attr * weight2
                        k = k + 1
            i += 1
        self.normalizeAttraction(attr_list)
        return attr_list

    def updateGridAttraction(self, attr_list, time):
        last = 2 * self.DIMENSION
        i = 0
        while i < 2 * self.DIMENSION + 1:
            attraction_decay1 = self.attraction_list[i]*(math.pow(self.decay_factor,(time -self.last_time_updated) ))
            attr_new = attr_list[i] + attraction_decay1
            if attraction_decay1 <= self.correlation_threshold and attr_new > self.correlation_threshold and i != last and not self.attributeChanged:
                self.setAttributeChanged(True)
            self.attraction_list[i] = attr_new
            i += 1

    def updateDecayedGridAttraction(self, time):
        i = 0
        while i < 2 * self.DIMENSION + 1:
            attraction_decay = self.attraction_list[i]*(math.pow(self.decay_factor,(time -self.last_time_updated) ))
            self.attraction_list[i] = attraction_decay
            i += 1

    def getAttractionAtIndex(self, i):
        """ generated source for method getAttractionAtIndex """
        return self.attraction_list[i]


# ============================================================================================================================================================================

class Clusterisation(object):
    """ generated source for class Clusterisation """
    gridList = {} 
    clusters = {}
    DIMENSION = 0
    DIMENSION_LOWER_RANGE = list()
    DIMENSION_UPPER_RANGE = list()
    DIMENSION_PARTITION = 0
    TOTAL_GRIDS = 1
    dense_threshold = 3.0

    #  Cm = 3.0
    sparse_threshold = 0.8

    #  Cl = 0.8
    time_gap = 0
    decay_factor = 0.998
    correlation_threshold = 0.0
    latestCluster = 0

    def __init__(self):
        self.DIMENSION = 0
        self.TOTAL_GRIDS = 1
        self.dense_threshold = 3.0
        self.sparse_threshold = 0.8
        self.time_gap = 0
        self.decay_factor = 0.998
        self.correlation_threshold = 0.0
        self.latestCluster = 0

    def printClusters(self):
        clusterKeys = self.clusters
        
        for ckey in clusterKeys:
            gridCoords = self.clusters.get(ckey)
            print " Cluster Index: " + ckey.__str__() 
            for coord in gridCoords:
                print "   Coordinates: " 
                print coord


    def get_dimension_info(self, data, first):
        dataList_1 = data.tolist()
        dataList = dataList_1[0]
        dimensionInfo = []
        for d in dataList:
            dimensionInfo.append(d)
        
        self.DIMENSION = N_DIM
        self.DIMENSION_PARTITION = PARTITION
        i =0
        
        while i < N_DIM:
            min_lower_range = dimensionInfo[i]

            max_upper_range = dimensionInfo[i]
            if first == 1:
                self.DIMENSION_LOWER_RANGE.append(min_lower_range)
                self.DIMENSION_UPPER_RANGE.append(max_upper_range)
            else:
                val1 = self.DIMENSION_LOWER_RANGE[i]
                val2 = self.DIMENSION_UPPER_RANGE[i]
                self.TOTAL_GRIDS = N_DIM*PARTITION
                
                if(min_lower_range < val1):
                    self.DIMENSION_LOWER_RANGE[i] = val1
                    
                if(max_upper_range > val2):
                    self.DIMENSION_UPPER_RANGE[i] = val2
            i += 1

        factor = 0.0
        pairs = 0.0
        j = 0
        total_pairs = 0
        while j < self.DIMENSION:
            factor = self.TOTAL_GRIDS / self.DIMENSION_PARTITION
            pairs = self.DIMENSION_PARTITION - 1
            total_pairs += (factor) * (pairs)
            j += 1

        self.correlation_threshold = self.dense_threshold / (total_pairs * (1 - self.decay_factor))


    def getNeighbours(self, from_):
        neighbours = list()
        dim = 0
        while dim < from_.getSize():
            val = from_.getDimension(dim)
            bigger = Coordinates(from_)
            bigger.setDimension(dim, val + 1)
            if self.gridList.has_key(bigger):
                neighbours.append(bigger)
            smaller = Coordinates(from_)
            smaller.setDimension(dim, val - 1)
            if self.gridList.has_key(smaller):
                neighbours.append(smaller)
            dim += 1
        return neighbours

    def getDimensionBigNeighbours(self, from_, dim):
        coord = Coordinates(from_)
        val = coord.getDimension(dim)
        bigger = Coordinates(from_)
        bigger.setDimension(dim, val + 1)

        if self.gridList.has_key(bigger):
            return bigger
        return coord

    def getDimensionSmallNeighbours(self, from_, dim):

        coord = Coordinates(from_)
        val = coord.getDimension(dim)
        smaller = Coordinates(from_)
        smaller.setDimension(dim, val - 1)

        if self.gridList.has_key(smaller):
            return smaller
        return coord

    def checkUnconnectedClusterAndSplit(self, clusterIndex):
        if not self.clusters.has_key(clusterIndex):
            return
        gridCoords = self.clusters[clusterIndex]
        grpCoords = {}
        if gridCoords.isEmpty():
            return
        dfsStack = Stack()
        dfsStack.push(gridCoords.iterator().next())
        while not dfsStack.empty():
            coords = dfsStack.pop()
            grid = self.gridList.get(coords)
            if grid.isVisited():
                continue 
            grid.setVisited(True)
            neighbours = getNeighbours(coords)
            for ngbr in neighbours:
                grpCoords.append(ngbr)
                dfsStack.push(ngbr)
        if len(grpCoords) == len(gridCoords):
            return
        newCluster = self.latestCluster + 1
        self.latestCluster += 1
        self.clusters[newCluster]= grpCoords
        for c in grpCoords:
            g = self.gridList.get(c)
            g.setCluster(newCluster)


    def findStronglyCorrelatedNeighbourWithMaxClusterSize(self, coord, onlyDense):
        resultCoord = Coordinates(coord)
        initCoord = Coordinates(coord)
        largestClusterSize = 0
        
        grid = self.gridList[initCoord]
        i = 0
        while i < self.DIMENSION:
            big_neighbour = self.getDimensionBigNeighbours(coord,i)
            
            small_neighbour = self.getDimensionSmallNeighbours(coord,i)

            if not big_neighbour.equals(initCoord):
                bigNeighbourGrid = self.gridList[big_neighbour]
                if not onlyDense or  bigNeighbourGrid.isDense():
                    bigNeighbourClusterIndex = bigNeighbourGrid.getCluster()

                    if not bigNeighbourClusterIndex == 0 and not bigNeighbourClusterIndex == grid.getCluster():

                        if bigNeighbourGrid.getAttractionAtIndex(2 * i + 1) > self.correlation_threshold and grid.getAttractionAtIndex(2 * i) > self.correlation_threshold:
                            if self.clusters.has_key(bigNeighbourClusterIndex):
                                bigNeighbourClusterGrids = self.clusters.get(bigNeighbourClusterIndex)
                                if len(bigNeighbourClusterGrids) >= largestClusterSize:
                                    largestClusterSize = len(bigNeighbourClusterGrids)
                                    resultCoord = big_neighbour

            if not small_neighbour.equals(initCoord):
                smallNeighbourGrid = self.gridList[small_neighbour]
                if not onlyDense or smallNeighbourGrid.isDense():
                    smallNeighbourClusterIndex = smallNeighbourGrid.getCluster()
                    if not smallNeighbourClusterIndex == 0 and not smallNeighbourClusterIndex == grid.getCluster():
                        if smallNeighbourGrid.getAttractionAtIndex(2 * i) > self.correlation_threshold and grid.getAttractionAtIndex(2 * i + 1) > self.correlation_threshold:
                            if self.clusters.has_key(smallNeighbourClusterIndex):        
                                smallNeighbourClusterGrids = self.clusters.get(smallNeighbourClusterIndex)
                                if len(smallNeighbourClusterGrids) >= largestClusterSize:
                                    largestClusterSize = len(smallNeighbourClusterGrids)
                                    resultCoord = small_neighbour
            i += 1
        return resultCoord






    def removeSporadicGrids(self, gridList, time):
        removeGrids = list()
        gridListKeys = gridList.keys()
        for glKey in gridListKeys:
            grid = gridList.get(glKey)
            lastTimeElementAdded = grid.getLastTimeElementAdded()
            densityThresholdFunc = (self.sparse_threshold * (1 - math.pow(self.decay_factor, time - lastTimeElementAdded + 1))) / (self.TOTAL_GRIDS * (1 - self.decay_factor))
            grid.updateDecayedGridDensity(time)
            grid.updateGridAttribute()
            grid.updateDecayedGridAttraction(time)
            grid.setLastTimeUpdated(time)
            if grid.getGridDensity() < densityThresholdFunc:
                removeGrids.append(key)
        for index in removeGrids:
            gridList.remove(index)   


    def adjustClustering(self, gridList, time):
        gridListKeys = gridList.keys()
        for coordkey in gridListKeys:
            grid = gridList.get(coordkey)
            key = coordkey.getCoords()

            if not grid.isAttributeChangedFromLastAdjust():
                continue 
            
            gridCluster = grid.getCluster()
            if grid.isSparse():
                if self.clusters.has_key(gridCluster):
                    clusterCoords = self.clusters.get(gridCluster)
                    grid.setCluster(0)
                    del clusterCoords[key]
                    self.checkUnconnectedClusterAndSplit(gridCluster)
            elif grid.isDense():
                neighbourCoords = self.findStronglyCorrelatedNeighbourWithMaxClusterSize(key, False);

                if not self.gridList.has_key(neighbourCoords) or neighbourCoords.equals(coordkey):

                    if not self.clusters.has_key(gridCluster):
                    
                        clusterIndex = self.latestCluster + 1
                        self.latestCluster += 1
                        coordset = []
                        coordset.append(key)
                        self.clusters[clusterIndex] = coordset
                        grid.setCluster(clusterIndex)
                    grid.setAttributeChanged(False)
                    continue

                neighbour = self.gridList.get(neighbourCoords)

                neighbourClusterIndex = neighbour.getCluster()
                if not self.clusters.has_key(neighbourClusterIndex):
                    continue 

                neighbourClusterGrids = self.clusters.get(neighbourClusterIndex)
                if neighbour.isDense():
                    if not self.clusters.has_key(gridCluster):
                        grid.setCluster(neighbourClusterIndex)
                        self.clusters[neighbourClusterIndex].append(key)
                    else:
                        currentClusterGrids = self.clusters.get(gridCluster)
                        size1 = 0
                        
                        for val in currentClusterGrids:
                            size1 +=1

                        size2 = 0
                        
                        for val in neighbourClusterGrids:
                            size2 +=1

                        if size2 >= size1:
                            for c in currentClusterGrids:
                                coord = Coordinates(c)
                                g = self.gridList.get(coord)
                                g.setCluster(neighbourClusterIndex)
                                self.clusters[neighbourClusterIndex].append(c)
                            del self.clusters[gridCluster]
                        else:
                            for c in neighbourClusterGrids:
                                g = self.gridList.get(c)
                                g.setCluster(gridCluster)
                                self.clusters[gridCluster].append(c)
                            del self.clusters[neighbourClusterIndex]
                elif neighbour.isTransitional():
                    if not self.clusters.has_key(gridCluster):
                        grid.setCluster(neighbourClusterIndex)
                        self.clusters[neighbourClusterIndex].append(key)
                    else:
                        currentClusterGrids = self.clusters.get(gridCluster)
                        if len(currentClusterGrids) >= len(neighbourClusterGrids):
                            self.clusters[gridCluster].append(neighbourCoords)
                            clusterGrid = clusters[neighbourClusterIndex]
                            del clusterGrid[neighbourCoords]
            elif grid.isTransitional():
                if self.clusters.has_key(gridCluster):
                    del self.clusters[gridCluster]
                neighbourCoords = self.findStronglyCorrelatedNeighbourWithMaxClusterSize(key, True);
                if not self.gridList.has_key(neighbourCoords) or neighbourCoords.equals(coordkey):
                    grid.setAttributeChanged(False)
                    grid.setCluster(0)
                    continue 
                
                neighbour = self.gridList.get(neighbourCoords)
                neighbourClusterIndex = neighbour.getCluster()
                if self.clusters.has_key(neighbourClusterIndex):
                    self.clusters[neighbourClusterIndex].append(key)
            grid.setAttributeChanged(False)


    def get_tseries_labels(self, data, t):
        dataList_1 = data.tolist()
        dataInfo = dataList_1
        
            
        datalength = len(dataInfo)
        if datalength != N_DIM:
            return
        grid_coords = list()
        data_coords = list()
        data = 0.0
        grid_Width = 0.0
        dim_index = 0
        i = 0
        while i < datalength:
            data = float(dataInfo[i])
            data_coords.append(data)
            if data >= self.DIMENSION_UPPER_RANGE[i] or data < self.DIMENSION_LOWER_RANGE[i]:
                return
            grid_Width = (self.DIMENSION_UPPER_RANGE[i] - self.DIMENSION_LOWER_RANGE[i]) / (DIMENSION_PARTITION)
            dim_index = int(math.floor((data - self.DIMENSION_LOWER_RANGE[i]) / grid_Width))
            grid_coords.append(dim_index)
            i += 1

        
        gridCoords = Coordinates(grid_coords)

        if not self.gridList.has_key(gridCoords):
            g = Grid(False,0,time,1,ATTRIBUTE.SPARSE, self.DIMENSION, self.DIMENSION_UPPER_RANGE, self.DIMENSION_LOWER_RANGE, self.DIMENSION_PARTITION, self.TOTAL_GRIDS, self.decay_factor, self.dense_threshold, self.sparse_threshold, self.correlation_threshold)
            attrL = g.getAttraction(data_coords, grid_coords)
            g.setInitialAttraction(attrL)
            self.gridList[gridCoords] = g
        else:
            g = self.gridList[gridCoords]
            gridCoords = None
            g.updateGridDensity(time)
            g.updateGridAttribute()
            attrL = g.getAttraction(data_coords, grid_coords)
            g.updateGridAttraction(attrL, time)
            g.setLastTimeUpdated(time)
            if time > 0:
                
                if time % self.time_gap == 0:
                    self.removeSporadicGrids(self.gridList, time)
                    self.adjustClustering(self.gridList, time)
            
  

if __name__ == '__main__':
    cluster = Clusterisation()
    centers = []
    centers = cluster_train_data(cluster)
    label_faces_from_video(centers)
