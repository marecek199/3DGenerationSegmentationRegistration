

import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import random
import pandas as pd
import open3d as o3d



from scipy.spatial.transform import Rotation as R
import h5py
import pickle as pck
import time
import math
from scipy.linalg import eigh
# from baldor.src.baldor import *



class Pose3D():
    
    def __init__(self, alpha=0, modelIdx=0, numberVotes=0) -> None:
        self.alpha_ = alpha
        self.modelIdx_ = modelIdx
        self.numberVotes_ = numberVotes
        self.voters_ = []
        self.pose_ = np.zeros((4,4))    

    def updatePose(self,newPose) -> None:
        self.pose_ = newPose 
        # self.omega_ = transform.to_axis_angle(newPose[:3,:3])[0]
        # self.angle_ = transform.to_axis_angle(newPose[:3,:3])[1]
        vRot = R.from_matrix(self.pose_[:3,:3])
        self.omega_ = vRot.as_rotvec()[:3]
        self.angle_ = vRot.as_rotvec()[2]
        self.q = vRot.as_quat()

    def updatePoseQuat(self,qNew):
        self.q = qNew
        oldPose = self.pose_
        vRotQ = R.from_quat(qNew)
        newPose = [vRotQ.as_matrix(), oldPose[:3,3],
                   [0,0,0,1]] 
        self.pose_ = newPose
        # self.omega_ = transform.to_axis_angle(newPose[:3,:3])[0]
        # self.angle_ = transform.to_axis_angle(newPose[:3,:3])[1]
        vrotM = R.from_matrix(newPose)
        self.omega_ = vrotM.as_rotvec()[:3]
        self.angle_ = vrotM.as_rotvec()[2]

    def updatePoseTranslation(self,t) -> None:
        self.pose_ = [[self.pose_[:2,:2], t],
                      [0,0,0,1]]

    def addVoter(self, voter) -> None:
        # self.voters_ = np.concatenate((self.voters_,voter))
        # self.voters_ = np.hstack((self.voters_,voter))
        self.voters_.append(voter)

    def updateScore(self,newScore) -> None:
        self.numberVotes_ = newScore




    

class PPF():

    samplingModelRelative_ = 0.03
    distanceRelative_ = 0.03
    angleRelative_ = 100

    angleInRadians_ = None
    alreadyTrained_ = False

    stepModelDistance_ = None
    sampledModelPointCloud_ = None
    hashTable_ = None
    referenceModelPointNumber_ = None
    listOfPoses_ = None
    stepModelDistance_ = None
    modelDiameter_ = None

    sceneDiameter_ = None
    referenceModelPointNumber_ = None

    
    def __init__(self, samplingRelative, distanceRelative, angleRelative) -> None:
        self.samplingModelRelative_ = samplingRelative
        self.distanceRelative_ = distanceRelative
        self.angleRelative_ = angleRelative
        self.angleInRadians_ = (360/self.angleRelative_)*np.pi/180

    def removeTraining(self) -> None:
        self.alreadyTrained_ = False
        self.hashTable_ = None 








    def trainModel(self, pointCloud ): # -> self: ?
        
        prepareStartTime = time.time()

        rangeX = [ np.min(np.asarray(pointCloud.points)[:,0]), np.max(np.asarray(pointCloud.points)[:,0])]
        rangeY = [ np.min(np.asarray(pointCloud.points)[:,1]), np.max(np.asarray(pointCloud.points)[:,1])]
        rangeZ = [ np.min(np.asarray(pointCloud.points)[:,2]), np.max(np.asarray(pointCloud.points)[:,2])]

        self.modelDiameter_ = np.sqrt( (rangeX[1]-rangeX[0])**2 + (rangeY[1]-rangeY[0])**2 + (rangeZ[1]-rangeZ[0])**2 )
        self.stepModelDistance_ = self.modelDiameter_ * self.samplingModelRelative_

        # self.sampledModelPointCloud_ = self.samplePCpoisson(copy.deepcopy(pointCloud), self.samplingModelRelative_ )
        if self.samplingModelRelative_ > 0:
            self.sampledModelPointCloud_ = self.downSampleToPoints(copy.deepcopy(pointCloud), self.samplingModelRelative_ )
        else:
            self.sampledModelPointCloud_ = np.hstack((np.asarray(pointCloud.points), np.asarray(pointCloud.normals)))
        self.referenceModelPointNumber_ = np.asarray(self.sampledModelPointCloud_).shape[0]
        self.minSamplDistanceModel = self.getMinSampleSize(pointCloud, stopIdx=self.referenceModelPointNumber_)
        
        print(f"Sampled model point cloud size : {self.referenceModelPointNumber_}")

        self.hashTable_ = dict()

        workingHashTable = copy.deepcopy(self.hashTable_)
        lamb = 0.98
        
        featureTime = []
        hashTime = []
        angleAlphaTime = []
        indexTime = []
        searchHashTime = []

        prepareEndTime = time.time() - prepareStartTime

        startTrainTime = time.time()
        # check all the point pairs in the point cloud -> (n over 2)
        for i,pc1 in enumerate(self.sampledModelPointCloud_):

            for j,pc2 in enumerate(self.sampledModelPointCloud_):

                # points must be different
                if(i is not j):
                    
                    timeLast = time.time()
                    F = self.featureVector(pc1,pc2)
                    featureTime.append(time.time() - timeLast)
                    
                    timeLast = time.time()
                    hash,hashKey = self.ppfHashing(F, self.angleInRadians_, self.stepModelDistance_)
                    hashTime.append(time.time() - timeLast)
                    
                    timeLast = time.time()
                    angleAlpha = self.computeAlpha(pc1,pc2)
                    angleAlphaTime.append(time.time() - timeLast)
                    
                    timeLast = time.time()
                    pc1ValDot = pc1[3:]
                    pc2ValDot = pc2[3:]
                    # dp = np.dot(pc1ValDot, pc2ValDot)
                    ppfIndex = i*self.referenceModelPointNumber_ + j
                    dp = pc1[3]*pc2[3] + pc1[4]*pc2[4] + pc1[5]*pc2[5]
                    valueVote = 1 - lamb * abs(dp)
                    node = [i, ppfIndex, angleAlpha, valueVote]
                    indexTime.append(time.time() - timeLast)                    


                    timeLast = time.time()
                    # if (node in self.workingHashTable):
                    if hash in workingHashTable:
                        # is_1d_array = isinstance(workingHashTable[hash], (list, np.ndarray)) and len(workingHashTable[hash]) > 0 and len(workingHashTable[hash][0]) == 1 
                        # if( is_1d_array ):
                        if( len(np.asarray(workingHashTable[hash],dtype=list).shape)==1 ):
                            workingHashTable[hash] += [node]
                        else:
                            workingHashTable[hash] += [node]
                    else:
                        workingHashTable[hash] = []
                        workingHashTable[hash] = [node]
                    searchHashTime.append(time.time() - timeLast)
                            
            if (i % 10 == 0):
                print(f"Trained : {round(100*i/self.referenceModelPointNumber_)} % ")

            # if ( i % (self.referenceModelPointNumber_ // 10) == 0 ):
            #     print(f"Trained : {round( 100*i/self.referenceModelPointNumber_)} %")
        
        
        print(f"\nTime ellapsed {time.time() - startTrainTime}\n")
        print(f"Mean of times :\nFeatureT {np.mean(featureTime)} s\nHashT {np.mean(hashTime)} s\nAlphaT {np.mean(angleAlphaTime)} s\nIndexT {np.mean(indexTime)} s\nSearchHashT {np.mean(searchHashTime)} s\n")
        # print(f"Max idxTime took {np.where(indexTime==np.max(indexTime))[0]}\n")

        print(f"Trained : {100} % ")
        self.hashTable_ = copy.deepcopy(workingHashTable)
        self.alreadyTrained_ = True
    












    def matchSceneModel(self,pointCloudScene, sceneFraction, recomputeScoreSwich, sceneSampling, poseFilterThreshold):
        global modelName, sceneName
        
        referencePointNumFunc = self.referenceModelPointNumber_
        angleRandiansFunc = self.angleInRadians_
        stepDistanceFunc = self.stepModelDistance_
        angleRelativeFunc = self.angleRelative_
        hasTableFunc = self.hashTable_
        sampledModelPointCloud = self.sampledModelPointCloud_

        averageVotingSwitch = False
        filterPosesSwich = False
        saveVotersSwitch = False

        sceneStep = int(np.trunc(1/sceneFraction))

        rangeX = [ np.min(np.asarray(pointCloudScene.points)[:,0]), np.max(np.asarray(pointCloudScene.points)[:,0])]
        rangeY = [ np.min(np.asarray(pointCloudScene.points)[:,1]), np.max(np.asarray(pointCloudScene.points)[:,1])]
        rangeZ = [ np.min(np.asarray(pointCloudScene.points)[:,2]), np.max(np.asarray(pointCloudScene.points)[:,2])]

        sceneDiameter = np.linalg.norm([(rangeX[1]-rangeX[0]), (rangeY[1]-rangeY[0]), (rangeZ[1]-rangeZ[0])])
        self.sceneDiameter_ = sceneDiameter

        stepSceneDistance = 1/(sceneDiameter * stepDistanceFunc)
        self.stepSceneDistance_ = stepSceneDistance

        self.stepSceneDistance_ = 1 / ( sceneDiameter / (self.modelDiameter_*sceneSampling))

        if sceneSampling>0:
            sampledScenePointCloud = self.downSampleToPoints(copy.deepcopy(pointCloudScene), sceneSampling )
        else:
            sampledScenePointCloud = np.hstack((np.asarray(pointCloudScene.points), np.asarray(pointCloudScene.normals)))            
        referenceScenePointNumber = np.asarray(sampledScenePointCloud).shape[0]

        print(f"Sampled scene point cloud size : {referenceScenePointNumber}")
         
        ## Expecting 3 hypotheses from each scene reference point ? ...
        # self.poseList = pd.DataFrame() ## TODO
        self.poseList = np.zeros(shape=(3*round( referencePointNumFunc / sceneStep ), 1))  
        # self.poseList = [] * 3*round( referencePointNumFunc / sceneStep )
        print(f"Scene points scanning : {referenceScenePointNumber/sceneStep}, PoseList : {self.poseList.shape[0]} ")
        self.poseList = []

        # obj.poseList=cell( 3*round(sceneSize/sceneStep) , 1)
        # disp(['scene ' num2str(sceneSize/sceneStep) ' poseList ' num2str(size(obj.poseList,1))])

        posesAdded = 0
        ## Vyber bod S_r z scenePointCloud 
        for i in range(0,referenceScenePointNumber,sceneStep):
            print(f"Matching : {i+1} of {referenceScenePointNumber}")

            # if i+1 == 261:
            #     axsa = 0

            pc1 = sampledScenePointCloud[i,:]

            ## Vytvor mu maticu na hlasovanie
            accumulator = np.zeros( ( (referencePointNumFunc+1)*angleRelativeFunc, 1), dtype="float")


            if saveVotersSwitch:
                coordAccumulator = []
                for saveVotesIdx in range((referencePointNumFunc+1)*angleRelativeFunc):
                    coordAccumulator.append([])
            

            ## Zisti jeho rotacne a translacne vlastnosti do axis -> x = 0
            rotate, translate = self.rotateTranslateMatrix(pc1)

            ## Selection of scene points
            ## Close points - TODO ?        
            #     ind=1:sceneSize
            # closePoints = ( sum(bsxfun(@minus,sampledScene(:,1:3),p1(:,1:3)).^2,2) < (obj.modelDiameter)^2 )
            # closePoints = np.sum((sampledScene[:, :3] - p1[:, :3])**2, axis=1)

            
            for j in range(referenceScenePointNumber):
                
                if i is not j:

                    pc2 = sampledScenePointCloud[j,:]

                    ## PPF medzi P1 a P2, vypocet transformacie p2 
                    F = self.featureVector(pc1, pc2)
                    hash,hashKey = self.ppfHashing(F, angleRandiansFunc, stepDistanceFunc)

                    pc2_transform = np.matmul( rotate, pc2[:3]) + translate

                    alphaScene = np.arctan2( -pc2_transform[2], pc2_transform[1]) 

                    if np.sin(alphaScene) * pc2_transform[2] > 0:
                        alphaScene *= -1

                    if hash == -2287278482449812728:
                        asdasudhgas = 0             
                    if i ==0 and j == 17:
                        asdsandsan=0       

                    ## Hladanie rovnakeho Hash v offline Hash tabulke
                    if (hash in self.hashTable_.keys()):

                        nodeList = self.hashTable_[hash]
                        nNodes = np.asarray(nodeList,dtype=object).shape[0]

                        if (len(np.asarray(nodeList,dtype=object).shape)==1):
                            nNodes = 1
                            nodeList = [nodeList, []]

                        # print(np.asarray(nodeList).shape)
                        # print(np.asarray(nodeList))
                        # adsadsj='das0das'

                        ## Ak je v tabulke viac "chlievikov" (moznosti pre dvojicu bodov) -> vsetky uloz do votovacej tabulky
                        for nodeInd in range(nNodes):
                            
                            modelI = nodeList[nodeInd][0]
                            ppfInd = nodeList[nodeInd][1]
                            alphaModel  = nodeList[nodeInd][2]

                            alphaAngle = alphaModel - alphaScene

                            if alphaAngle > np.pi:
                                alphaAngle = alphaAngle - 2*np.pi
                            elif alphaAngle < - np.pi:
                                alphaAngle = alphaAngle + 2*np.pi

                            ## Dopocitanie predpocitaneho uhlu alfa z tabulky 
                            alphaIndex = self.normal_round( (angleRelativeFunc-1) * (alphaAngle+np.pi) / (2*np.pi))

                            accuIndex = (modelI+1) * angleRelativeFunc + alphaIndex-1 ############### MODELI + 1??????

                            ###################
                            #  ADAPTIVE VOTING IN ACCMULUATOR
                            if not averageVotingSwitch : 
                                voteVal = 1
                            else:                                
                                voteVal = nodeList[nodeInd][3]
                            
                            ## Accumulator Matrix -> ukladanie h;aspv -> riadky() * stlpce  
                            accumulator[accuIndex] += voteVal

                            if accuIndex == 50:
                                asdsadnf = 0
                            
                            if saveVotersSwitch:                                
                               coordAccumulator[accuIndex].append([j, ppfInd])

            
            ######################
            #selection of the poses from accumulator
            ######################
            #version 1
            #return all poses with more than 95% of votes of the best pose
            accuMax = 0.95*np.max(accumulator)                          ## =67.45 <=> model=NN_100pts; scene=NN_50pts
            accuMaxInd = accumulator > accuMax

            # Vyratanie POSE Estimation -> pre vsetky chlievku v 'Accumulator Matrix' kde je aspon 95% maxima hlasov; pridanie do PoseListu
            for poseIdx,potentialPose in enumerate(accuMaxInd):                   
                
                if potentialPose:
                    
                    alphaMaxInd = (poseIdx+1) % angleRelativeFunc 
                    iMaxInd = int(( (poseIdx+1) - alphaMaxInd ) / angleRelativeFunc )
                    
                    iRotate = rotate.transpose()
                    iTranslate = np.matmul( iRotate, translate)

                    iRTMatrix = np.eye(4)
                    iRTMatrix[:3,:3] = iRotate
                    iRTMatrix[:3,3] = -iTranslate

                    pMax = sampledModelPointCloud[iMaxInd-1,:]

                    rotateMax, translateMax = self.rotateTranslateMatrix(pMax)
                    tMax = np.eye(4)
                    tMax[:3,:3] = rotateMax
                    tMax[:3,3] = translateMax

                    alphaAngle = (2*np.pi) * (alphaMaxInd) / (angleRelativeFunc-1) - np.pi
                    # alphaAngle = (2*np.pi) * (alphaMaxInd+1) / (angleRelativeFunc) - np.pi

                    rotvectX = R.from_rotvec([alphaAngle,0,0])
                    translateAlpha = np.eye(4)
                    translateAlpha[:3,:3] = rotvectX.as_matrix()
                    # translateAlpha = self.XrotateMatrix(alphaAngle)


                    translatePose = np.matmul( np.matmul( iRTMatrix, translateAlpha), tMax)
                    #%normalization of the votes from accumulator                
                    #%version 1
                    #%no normalization
                    numVotes = accumulator[poseIdx]                    
                    #%version 2
                    #%normalize number of votes by number of tested
                    #%point-pair features in scene
                    #%numVotes=accumulator(peak,1)/sum(closePoints)

                    # modelPointCloudTest = o3d.io.read_point_cloud(modelName)
                    # scenePointCloudTest = o3d.io.read_point_cloud(sceneName)
                    # transformationResult = np.eye(4)
                    # transformationResult[:, :] = translatePose
                    # modelPointCloudTest.transform(transformationResult)
                    # o3d.visualization.draw_geometries([modelPointCloudTest+scenePointCloudTest])
                    

                    if posesAdded==21:
                        dasdkashbjdas=45121
                    
                    ## MOZNO CHYBA S MODELIDX ? -> posesAdded?
                    newPose = Pose3D( alpha=alphaAngle, modelIdx=posesAdded, numberVotes=numVotes)
                    newPose.updatePose(translatePose)

                    if saveVotersSwitch:
                        voted = np.asarray(coordAccumulator[poseIdx],dtype=object)
                        # % compute coordinates of the voters
                        # % i scene  j scene  i model j model
                        # modelI = np.floor(voted[:, 1] / self.referenceModelPointNumber_) + 1
                        # modelY = (voted[:, 1] % self.referenceModelPointNumber_ ) + 1
                        # sampledScene_i = np.tile(sampledScenePointCloud[i, 0:6], (voted.shape[0], 1))
                        # sampledScene_voted = sampledScenePointCloud[list(voted[:, 0])]
                        # obj_sampledPC_modelI = self.sampledModelPointCloud_[modelI]
                        # obj_sampledPC_modelY = self.sampledModelPointCloud_[list(modelY)]
                        # votes = np.concatenate((sampledScene_i, sampledScene_voted, obj_sampledPC_modelI, obj_sampledPC_modelY), axis=1)

                        modelI = np.trunc(voted[:, 1] / self.referenceModelPointNumber_) + 1
                        modelY = np.remainder(voted[:, 1], self.referenceModelPointNumber_) + 1
                        votes = np.concatenate((
                            np.tile( sampledScenePointCloud[i, :6], (voted.shape[0], 1)),
                            sampledScenePointCloud[voted[:, 0], :6],
                            self.sampledModelPointCloud_[modelI, :6],
                            self.sampledModelPointCloud_[modelY, :6]
                        ), axis=1)

                        newPose.addVoter(votes)
                    else:
                        newPose.addVoter(0)

                    self.poseList.append(newPose)
                    # self.poseList[posesAdded] = newPose
                    posesAdded += 1

        ## Remove empty cells
        print(f"Number of pose : {len(self.poseList)}")

        # votes = [ v.numberVotes_ for v in self.poseList]
        # voteMostIdx = np.where( votes == np.max([ v.numberVotes_[0] for v in self.poseList]))[0][0]
        # print(f"Final max voted rotation : \n{self.poseList[voteMostIdx].pose_}")

        # %if pose filtering enabled
        if poseFilterThreshold > 0 :
            self.filterPoses(poseFilterThreshold)
            # print(f"Filtered posses : {self.poseList}")
        
        clustered, votesClustered = self.clusterPoses()
        # print(f"Clustered poses : {np.asarray(clustered,dtype=object).shape[0]}")

        averageClust = self.averageClusters(clustered,votesClustered)
        # averageClustQuat = self.averageClustersQuat(clustered, votesClustered)
        resultOrig = self.sortPoses(averageClust)
        # print(f"Result after sort & average poses \n{resultOrig[0].pose_}")

        # recomputeScoreSwich=True
        if recomputeScoreSwich:
            # recomputed = self.recomputeScore(resultOrig, scenePointCloud)
            recomputed = self.recomputeScoreScene(resultOrig, sampledScenePointCloud)
            result = self.sortPoses(recomputed)
            resultOrig = result
        else:
            pass
            # result = resultOrig
            # resultOrig = result

        # return resultOrig[0].pose_, self.poseList[voteMostIdx].pose_, clustered
        return clustered, resultOrig

            



    def avg_quaternion_markley(self,Q):
        # Form the symmetric accumulator matrix
        A = np.zeros((4,4))
        M = Q.shape[0]

        for i in range(M):
            q = Q[i,:].reshape(-1,1)
            A += q.T   # rank 1 update

        # scale
        A /= M

        # Get the eigenvector corresponding to largest eigenvalue
        Eval, Qavg = eigh(A, eigvals=(3,3))  # largest eigenvalue

        return Qavg.flatten()


    def TransformPointCloud(self,modelPC, transformation):        
        pointCloudXYZ = copy.deepcopy(modelPC)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointCloudXYZ[:,:3])
        pcd.normals = o3d.utility.Vector3dVector(pointCloudXYZ[:,3:])

        transformationResult = np.eye(4)
        transformationResult[:, :] = transformation
        pcd.transform(transformation)

        pointCloudOut = np.array( pcd.points )
        return pointCloudOut


    def TransformPoses(self, model, actPose):
        
        newPc = copy.deepcopy(model)
        # newPc = pc.copy()

        p = np.dot(actPose, np.hstack((model[:,:3], np.ones((model.shape[0], 1)))).T).T

        newPc[:,:3] = p[:,:3] / p[:,3][:,np.newaxis]

        # with normals
        if model.shape[1] > 3:
            n = np.dot(actPose[:3,:3], model[:,3:6].T).T
            nNorm = np.sqrt(np.sum(n**2, axis=1))
            indx = (nNorm > 0)

            n[indx,0] /= nNorm[indx]
            n[indx,1] /= nNorm[indx]
            n[indx,2] /= nNorm[indx]

            newPc[:,3:6] = n
            
        return newPc
    



    def scoreRestore(self, poses, scene):
        pass
        #TODO
        

    def averageClusters(self, clusters, votesClustered):
        global modelName, sceneName
        resultPoses = []
        for i,actualClus in enumerate(clusters):
            sum = np.zeros((4,4))
            count = 0
            allVoters=[]
            for actualPose in actualClus:
                pose = actualPose.pose_
                sum += pose
                count += 1        
                allVoters.append(actualPose.voters_)
            finalClustAverPose = sum / count 

            newPose = Pose3D(0,-1, votesClustered[i])
            newPose.updatePose(finalClustAverPose)
            newPose.addVoter(np.asarray(allVoters,list).flatten())
            resultPoses.append(newPose)

            # newPose.updatePoseTranslation(finalClustAverPose[:3,3])
            # newPose.updatePoseQuat(self.avg_quaternion_markley(clusters))
            # newPose.addVoter(votesClustered[i])

            # modelPointCloudTest = o3d.io.read_point_cloud(modelName)
            # scenePointCloudTest = o3d.io.read_point_cloud(sceneName)
            # transformationResult = np.eye(4)
            # transformationResult[:, :] = finalClustAverPose
            # modelPointCloudTest.transform(transformationResult)
            # o3d.visualization.draw_geometries([modelPointCloudTest+scenePointCloudTest])
            # dsaihdsaj = 0

            # if len(clusters)>1:
                # resultPoses.append(newPose)
            # else:
            #     resultPoses = newPose

        return resultPoses
    
    def averageClustersQuat(self, clusters, votes):
        # resultPoses averages the poses in each cluster
        resultPoses = [None] * len(clusters)
        for i in range(len(clusters)):
            newPose = Pose3D(0, -1, votes[i])
            # get all quaternions
            qs = np.array([x.q for x in clusters[i]])
            # get all translations
            ts = np.array([x.pose_[0:3, 3] for x in clusters[i]])
            # get all voters
            voters = np.hstack([x.voters_ for x in clusters[i]])
            newPose.updatePoseTranslation(np.mean(ts, axis=1))
            newPose.updatePoseQuat(self.avg_quaternion_markley(qs.T).T)
            newPose.addVoter(voters)
            resultPoses[i] = newPose
        return resultPoses    

    def argBubbleSort(self,arr):
        n = len(arr)
        argSort = list(range(n))
        # Traverse through all array elements
        for i in range(n):
            # Last i elements are already in place
            for j in range(0, n-i-1):
                # traverse the array from 0 to n-i-1
                # Swap if the element found is greater
                # than the next element
                if arr[j] < arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
                    argSort[j], argSort[j+1] = argSort[j+1], argSort[j]
        return argSort
    
    def sortPoses(self,poses):
        # % sortPoses will sort the poses according to their number of votes

        numVotes = [p.numberVotes_[0] for p in poses]

        # sortIdx = np.argsort(numVotes)[::-1]
        # sortIdx = np.argsort(-1*np.array(numVotes))
        sortIdx = np.array(self.argBubbleSort(numVotes))

        if sortIdx.shape[0]>1:        
            sorted = np.asarray(poses)[sortIdx]
        else:
            sorted = poses
        return sorted
    

    def poseSorting(self):
        pass
        #TODO


    def filterPoses(self, poseFilterThreshold) : 
        # % filterPoses will prune the hypotheses according to the poseFilterThreshold
        # %select which poses to remove 
        ind = [x.numberVotes_ < poseFilterThreshold for x in self.poseList]
        self.poseList[ind] = []
        pass


    # def recomputeScore(self,poses,scene):
    #         global modelName, sceneName
    #         # % recomputeScore will calculate the matching score for pose list
        
    #         outPoses = copy.deepcopy(poses)
            
    #         model = self.sampledModelPointCloud_
            
    #         # rScene = self.stepSceneDistance_ * self.sceneDiameter_
            
    #         # rSceneSquare = rScene**2
    #         rMinSampleModel = self.minSamplDistanceModel

    #         # %for each pose
    #         # for i=1:size(poses,1)
    #         for i in range(len(poses)):
            
    #             # curMod = self.TransformPoses( model, poses[i].pose_)
    #             curMod = self.TransformPointCloud( model, poses[i].pose_) ### TRANSFORMING ???
                
    #             score=0
    #             distArr = []
    #             # %check each point on the model
    #             for j in range(len(curMod)):
                    
    #                 #!!!!!!!!!!!!!!
    #                 ### HERE IS A PROBLEM, DOESNT REFLECT THE DISTANCE EVEN THOUGH THE IDEA ITSELF SEEMS GOOD
    #                 dist = np.sum( np.abs(np.asarray(scene.points,dtype=object)[:,:3] - np.ones(np.asarray(scene.points).shape)*curMod[j,:3]), axis=1 ) < rMinSampleModel * 70

    #                 #########
    #                 # if (j==0) or (j==1 and i==0) :
    #                 # # if True:
    #                 #     scenePointCloudTest = o3d.io.read_point_cloud(sceneName)
    #                 #     mask = np.zeros(np.asarray(scenePointCloudTest.points).shape[0], dtype=bool)
    #                 #     mask[:] = True
    #                 #     colors = np.zeros((np.asarray(scenePointCloudTest.points).shape[0], 3))
    #                 #     colors[mask] = [ 0.0, 0.0, 0.0]
    #                 #     scenePointCloudTest.colors = o3d.utility.Vector3dVector(colors)
                            
    #                 #     modelPointCloudTest = o3d.io.read_point_cloud(modelName)
    #                 #     transformationResult = np.eye(4)
    #                 #     transformationResult[:, :] = poses[i].pose_
    #                 #     modelPointCloudTest.transform(transformationResult)
    #                 #     mask = np.zeros(np.asarray(modelPointCloudTest.points).shape[0], dtype=bool)
    #                 #     mask[:] = True
    #                 #     colors = np.zeros((np.asarray(modelPointCloudTest.points).shape[0], 3))
    #                 #     colors[mask] = [0.0, 0.0, 1.0]
    #                 #     modelPointCloudTest.colors = o3d.utility.Vector3dVector(colors)

    #                 #     pcdPoint = o3d.geometry.PointCloud()
    #                 #     pcdPoint.points = o3d.utility.Vector3dVector(np.array(curMod[j:j+1,:3]))
    #                 #     mask = np.zeros(np.asarray(pcdPoint.points).shape[0], dtype=bool)
    #                 #     mask[:] = True
    #                 #     colors = np.zeros((np.asarray(pcdPoint.points).shape[0], 3))
    #                 #     colors[mask] = [ 1.0, 0.0, 0.0]
    #                 #     pcdPoint.colors = o3d.utility.Vector3dVector(colors)

    #                 #     o3d.visualization.draw_geometries([modelPointCloudTest + scenePointCloudTest + pcdPoint])

    #                 if sum(dist) > 0:
    #                     score += 1

    #             # print(f"Score {i}: {score}, {score/len(curMod)}")
    #             outPoses[i].updateScore([score / len(curMod)])
    #             # outPoses[i].updateScore([score])
            
    #         return outPoses
            


    def recomputeScoreScene(self,poses,scene):
            global modelName, sceneName
            # % recomputeScore will calculate the matching score for pose list
        
            outPoses = copy.deepcopy(poses)
            model = self.sampledModelPointCloud_
            
            # rScene = self.stepSceneDistance_ * self.sceneDiameter_
            
            # rSceneSquare = rScene**2
            rMinSampleModel = self.minSamplDistanceModel

            # %for each pose
            # for i=1:size(poses,1)
            for i in range(len(poses)):
            
                # curMod = self.TransformPoses( model, poses[i].pose_)
                curMod = self.TransformPointCloud( model, poses[i].pose_) ### TRANSFORMING ???
                
                score=0
                distArr = []
                # %check each point on the model
                # for j in range(np.array(scene.points).shape[0]):
                for j in range( scene.shape[0]):
                    
                    #!!!!!!!!!!!!!!
                    ### HERE IS A PROBLEM, DOESNT REFLECT THE DISTANCE EVEN THOUGH THE IDEA ITSELF SEEMS GOOD
                    dist = np.sum( np.abs( curMod[:,:3] - np.ones(curMod.shape)*scene[j,:3]  ), axis=1 ) < rMinSampleModel * 40

                    #########
                    # if (j==0) or (j==1 and i==0) :
                    # # if True:
                    #     scenePointCloudTest = o3d.io.read_point_cloud(sceneName)
                    #     mask = np.zeros(np.asarray(scenePointCloudTest.points).shape[0], dtype=bool)
                    #     mask[:] = True
                    #     colors = np.zeros((np.asarray(scenePointCloudTest.points).shape[0], 3))
                    #     colors[mask] = [ 0.0, 0.0, 0.0]
                    #     scenePointCloudTest.colors = o3d.utility.Vector3dVector(colors)
                            
                    #     modelPointCloudTest = o3d.io.read_point_cloud(modelName)
                    #     # modelPointCloudTest = self.downSampleToPoints( modelPointCloudTest, self.stepModelDistance_)
                    #     modelPointCloudTest = modelPointCloudTest.voxel_down_sample( voxel_size=self.samplingModelRelative_ )
                    #     transformationResult = np.eye(4)
                    #     transformationResult[:, :] = poses[i].pose_
                    #     modelPointCloudTest.transform(transformationResult)
                    #     mask = np.zeros(np.asarray(modelPointCloudTest.points).shape[0], dtype=bool)
                    #     mask[:] = True
                    #     colors = np.zeros((np.asarray(modelPointCloudTest.points).shape[0], 3))
                    #     colors[mask] = [0.0, 0.0, 1.0]
                    #     modelPointCloudTest.colors = o3d.utility.Vector3dVector(colors)

                    #     pcdPoint = o3d.geometry.PointCloud()
                    #     arr = []
                    #     pointAct = np.asarray(scene.points,dtype=object)[j:j+1,:3]
                    #     for i in range(-1,2):
                    #         for j in range(-1,2):
                    #             for k in range(-1,2):
                    #                 arr.append([pointAct[0][0]+i*0.1, pointAct[0][1]+j*0.1, pointAct[0][2]+k*0.1])
                    #     arr = np.array(arr)

                    #     pcdPoint.points = o3d.utility.Vector3dVector(np.array( arr ) )
                    #     mask = np.zeros(np.asarray(pcdPoint.points).shape[0], dtype=bool)
                    #     mask[:] = True
                    #     colors = np.zeros((np.asarray(pcdPoint.points).shape[0], 3))
                    #     colors[mask] = [ 1.0, 0.0, 0.0]
                    #     pcdPoint.colors = o3d.utility.Vector3dVector(colors)
                    #     o3d.visualization.draw_geometries([modelPointCloudTest + scenePointCloudTest + pcdPoint])

                    if sum(dist) > 0:
                        score += 1

                # print(f"Score {i}: {score}, {score / scene.shape[0]}")
                outPoses[i].updateScore([score / scene.shape[0] ])
                # outPoses[i].updateScore([score])
            
            return outPoses



    def getMinSampleSize(self,pointCloudModel,stopIdx=None):
        '''Gets the smallets distance between two points in model pointCloudModel'''
        pointCloud = copy.deepcopy( np.array(pointCloudModel.points) )
        minDistFound = self.modelDiameter_

        for i,_ in enumerate(pointCloud):
            pc1 = pointCloud[i]
            for j,_ in enumerate(pointCloud):
                pc2 = pointCloud[j]
                if i != j:
                    actualDistance = np.linalg.norm([pc1[0]-pc2[0], pc1[1] - pc2[1], pc1[2] - pc2[2]])
                    if actualDistance < minDistFound:
                        minDistFound = actualDistance
            if stopIdx:
                if i > 50: break
                # if i > stopIdx: break
        return minDistFound


    def clusterPoses(self): 
        modelDiameterLoc = self.modelDiameter_
        angleRadiansLoc = self.angleInRadians_
        
        #sort poses by number of votes
        sorted = self.sortPoses(self.poseList)
        # sorted = self.poseList
        
        poseClusters = []
        # poseClusters = np.zeros((np.asarray(sorted).shape[0],1))
        votes = np.zeros((np.asarray(sorted).shape[0],1))
        
        clusterNum = 0
        
        #search all poses
        for i  in range(len(sorted)):
            
            assigned = False
            curPose = sorted[i]
            
            for j in range(clusterNum):

                if len(np.asarray(poseClusters[j]).shape) == 1:
                    poseCenter = poseClusters[j][0]
                else:
                    poseCenter = poseClusters[j]
                
                # if self.comparePosesMex( curPose.pose_[:3,3], poseCenter.pose_[:3,3], curPose.angle_, poseCenter.angle_, curPose.omega, poseCenter.omega_, modelDiameterLoc,angleRadiansLoc ):                    
                if( self.comparePoses( curPose, poseCenter,modelDiameterLoc,angleRadiansLoc) ):
                    # if type(poseClusters[j]) != list:
                    #     poseClusters[j] = [poseClusters[j],curPose]
                    # else:
                    poseClusters[j].append(curPose)
                    # poseClusters[j].append([curPose])
                    votes[j] += curPose.numberVotes_[0]
                    assigned = True
                    break

            if not assigned:                
                poseClusters.append([curPose])
                votes[clusterNum] = curPose.numberVotes_[0]
                clusterNum += 1

        # clusters = [ poseClusters[1:clusterNum] ]
        clusters = poseClusters[:]
        votes = votes[:clusterNum]

        return (clusters, votes)




    # def comparePosesMex(self,pose1, pose2, objectDiameter, angleLimit):
    #     # translation
    #     d = np.linalg.norm( pose1.pose_ - pose2.pose_)

    #     # angle
    #     phi = np.abs(pose1.angle_ - pose2.angle_)

    #     if( d < (0.1 * objectDiameter) and phi < angleLimit ):
    #         resultOut = True
    #     else:
    #         resultOut = False




    def comparePoses(self,pose1, pose2, objectDiameter, angleLimit):
        # translation
        d = np.linalg.norm( pose1.pose_[:3,3] - pose2.pose_[:3,3] )
        # angle
        phi = np.abs(pose1.omega_ - pose2.omega_)

        if( d < (0.1 * objectDiameter) and np.all(phi < angleLimit) ):
            resultOut = True
        else:
            resultOut = False

        return resultOut












    def rotateTranslateMatrix(sellf, point):
        """Transformation computing to rotate point to origin & n1 -> x axis
        from the given equation"""
        # axis = [0, point[5], -point[4]]
        # axis_norm = np.linalg.norm(axis)
        # if axis_norm:
        #     axis /= axis_norm
        # else:
        #     axis = [0, 1, 0]
        # rotationVector = np.arccos(point[3]) * axis
        # r = R.from_rotvec(rotationVector)
        # Rotate = r.as_matrix()
        # translate = -np.dot(Rotate, point[:3])
        
        ## NOT IMPROVED
        angle = np.arccos(point[3])
        axis = [0, point[5], -point[4]]
        if ( point[4] == 0 and point[5] == 0 ):
            axis = [0, 1, 0] 
        else:
            axisNorm = np.linalg.norm(axis)
            ## Err - iba ak je vacsi ako 0>
            if axisNorm>0 : axis = axis/axisNorm
        rotationVector = angle * axis
        r = R.from_rotvec(rotationVector)
        Rotate = r.as_matrix()
        translate = np.matmul(-Rotate, point[:3].T)
        # Err - vstupnu vektor?

        return Rotate, translate


    def computeAlpha(self, point1,point2) -> float:
        """ Estimate alpha difference angle between 2 points, used in voting matrix A """
        # rotate, translate = self.rotateTranslateMatrix(point1)
        # mpt = rotate @ point2[:3] + translate
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     alpha = np.arctan(-mpt[2] / mpt[1])        
        # # alpha = np.arctan(-mpt[2] / mpt[1])
        # if np.sign(np.sin(alpha) * mpt[2]) < 0:
        #     alpha *= -1

        ## NOT IMPROVED
        rotate, translate = self.rotateTranslateMatrix(point1)
        mpt = rotate.dot(point2[:3].T) + translate
        # mpt = np.matmul( rotate, point2[:3].T ) + translate
        alpha = np.arctan2( -mpt[2], mpt[1])
        if( np.sin(alpha)*mpt[2]>0 ): alpha *= -1

        return alpha


    def ppfHashing(self, F, angleRad, stepDist) :
        """ Hash the given F with one value """

        hashKey = [int(np.trunc(x/angleRad)) for x in F[:3]]
        hashKey.append(int(np.trunc(F[3]/stepDist))) 

        # hashFinal = mmh3.hash(hashKey, 42, signed=False) # returns a 32-bit unsigned int
        hashFinal = hash(tuple(hashKey))
        return (hashFinal,hashKey)
    

    def featureVector(self, point1, point2) -> np.ndarray:
        """ Calculate the F vector for given points """
        # d = point2[:3] - point1[:3]
        # dLength = np.linalg.norm(d)
        # if dLength > 0:
        #     dNorm = d / dLength
        #     p1 = np.concatenate((dNorm[:, np.newaxis], dNorm[:, np.newaxis], point1[3:, np.newaxis]), axis=1)
        #     p2 = np.concatenate((point1[3:, np.newaxis], point2[3:, np.newaxis], point2[3:, np.newaxis]), axis=1)
        #     cross = np.cross(p1, p2)
        #     dot = np.dot(p1.T, p2)
        #     angle = np.arctan2(np.linalg.norm(cross, axis=0), dot.diagonal())
        #     F = np.concatenate((angle, np.array([dLength])))
        # else:
        #     F = np.asarray([0, 0, 0, dLength])


        ## NOT IMPROVE
        ##
        n1 = point1[3:]
        n2 = point2[3:]
        ##
        d = (point2[:3] - point1[:3])
        dLength = np.linalg.norm(d)
        if (dLength>0):
            dNorm = d / dLength
            f1 = np.arctan2(np.linalg.norm(np.cross(dNorm, n1)), np.dot(dNorm, n1))
            f2 = np.arctan2(np.linalg.norm(np.cross(dNorm, n2)), np.dot(dNorm, n2))
            f3 = np.arctan2(np.linalg.norm(np.cross(n1, n2)), np.dot(n1, n2))
            F=[f1,f2,f3,np.linalg.norm(d)]
            # p1 = np.array((dNorm, dNorm, np.transpose(point1[3:])) ).T
            # p2 = np.array((point1[3:], point2[3:], np.transpose(point2[3:])) ).T
            # F = np.array( ( *np.arctan2( np.sqrt(np.sum((np.cross(p1,p2))**2)), np.array(( np.dot(p1[:,0],p2[:,0]), np.dot(p1[:,1],p2[:,1]), np.dot(p1[:,2],p2[:,2]) ))  ) , np.linalg.norm(d) ) )
        else:
            F = np.asarray([0,0,0,dLength])
            
        return F

            
    def downSampleToPoints(self, pointCloud, step) -> np.ndarray: 
        """Sample down the given point point cloud and stepDistane 
        compute new normals if needed"""

        # Recalcul normals, dorobit. Mozna chyba?
        recalculNormals=False
        if recalculNormals:
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pointCloud[:,:3])
            pointCloudOut = pointCloud.voxel_down_sample( voxel_size=step )
            pointCloudOut = np.asarray(pointCloudOut.points)
            print("Normal recalcul not done yet..")
        else:
            pointCloudOut = pointCloud.voxel_down_sample( voxel_size=step )
            pointCloudOut = np.hstack((np.asarray(pointCloudOut.points), np.asarray(pointCloudOut.normals)))

        return pointCloudOut


    # def samplePcPoisson(self,pc,step) -> np.ndarray:
    def samplePCpoisson(self, pointCloud, sampleStep, refitNormals=False) -> np.ndarray:
        """
        Sample the point cloud using the poisson disc sampling
        Sample step is given relative to the point cloud
        If third argument present, normal vectors are recomputed
        """
        # pc = np.asarray(pointCloud.points)
        normals = np.asarray(pointCloud.normals)
        pc = np.hstack((np.asarray(pointCloud.points), np.asarray(pointCloud.normals)))

        rangeX = [np.min(pc[:,0]), np.max(pc[:,0])]
        rangeY = [np.min(pc[:,1]), np.max(pc[:,1])]
        rangeZ = [np.min(pc[:,2]), np.max(pc[:,2])]

        dx = rangeX[1] - rangeX[0]
        dy = rangeY[1] - rangeY[0]
        dz = rangeZ[1] - rangeZ[0]

        # length of diagonal of bounding box
        d = np.sqrt(dx**2 + dy**2 + dz**2)

        # minimal distance of the points
        r = d * sampleStep

        rs = r**2

        boxSize = r / np.sqrt(3)

        samplesInDimX = int(np.floor(dx / boxSize))
        samplesInDimY = int(np.floor(dy / boxSize))
        samplesInDimZ = int(np.floor(dz / boxSize))

        map_ = np.zeros((samplesInDimX, samplesInDimY, samplesInDimZ))
        counter = 0

        for i in range(pc.shape[0]):
            xCell = int(np.floor(samplesInDimX * (pc[i, 0] - rangeX[0])) / dx)
            yCell = int(np.floor(samplesInDimY * (pc[i, 1] - rangeY[0])) / dy)
            zCell = int(np.floor(samplesInDimZ * (pc[i, 2] - rangeZ[0])) / dz)

            # select neighbors 5x5x5 points
            neigh = map_[max((xCell-2), 0):min((xCell+3), map_.shape[0]), max((yCell-2), 0):min((yCell+3), map_.shape[1]), max((zCell-2), 0):min((zCell+3), map_.shape[2])].reshape(-1)

            # no points around
            if np.all(neigh == 0):
                map_[xCell,yCell,zCell] = i
                counter+=1

            # check distance
            elif np.all(np.sum((pc[np.nonzero(neigh)[0][0],:3] - pc[i,:3])**2) >= rs):
                map_[xCell,yCell,zCell] = i
                counter+=1

        indx = np.nonzero(map_)[0]
        sampledPC = pc[indx,:]

        if refitNormals:
            windowWidth = (5 * boxSize)**2

            if sampledPC.shape[1] < 4:
                sampledPC = np.hstack((sampledPC, np.zeros(sampledPC.shape)))

            for i in range(sampledPC.shape[0]):
                # compute distance of the point i to all other points - sqrt not needed
                dist2 = np.sum((sampledPC[:,:3] - sampledPC[i,:3])**2, axis=1)

                # get indexes of nearest 10 points
                elems = np.partition(dist2, 15)[:15]
                indxs = np.argpartition(dist2, 15)[:15]

                neigh = indxs[elems < windowWidth]

                if len(neigh) > 5:
                    v,_,_ = self.affine_fit(sampledPC[neigh,:3]).T
                    if v[2] > 0:
                        sampledPC[i,3:6] = v
                    else:
                        sampledPC[i,3:6] = -v

        if pc.shape[1] > 3:
            nNorm = np.sqrt(np.sum(sampledPC[:,3:6]**2, axis=1))
            indx = nNorm > 0
            sampledPC[indx,3:6] = sampledPC[indx,3:6] / nNorm[indx,np.newaxis]

        return sampledPC

    def affine_fit(X):
        p = np.mean(X, axis=0)

        R = X - p

        V,D = np.linalg.eig(R.T @ R)
        n = V[:,0]
        V = V[:,1:]

        return n,V,p

    def normal_round(self,n):
        if n - math.floor(n) < 0.5:
            return math.floor(n)
        return math.ceil(n)


    def savePPFTrainedModel(self,ppfObject) -> True:
        with open(f"PPF_models/ppfModelSaved_{ppfObject.referenceModelPointNumber_}.pkl", 'wb') as outp:        
            pck.dump(ppfObject, outp, pck.HIGHEST_PROTOCOL)
            print(f"Saved : PPF_models/ppfModelSaved_{ppfObject.referenceModelPointNumber_}.pkl")
        return True


    def loadPPFTrainedModel(self,name) -> object:
        with open("PPF_models/"+name, 'rb') as inp:
            ppf = pck.load(inp)
        return ppf





if __name__=="__main__":
    
    loadModel = True
    # loadModel = False

    # modelName = "data/IPAGearShaft_100pts.ply"
    modelName = "data/T_join.ply"
    # modelName = "data/T_join_Surface.ply"
    # modelName = "data/firstNNtest_8_100pts.ply"

    modelPointCloud = o3d.io.read_point_cloud(modelName)
    ppf = PPF(5,-1,100)
    
    if not loadModel:
        # ppf = PPF(-1,-1,30)
        ppf.trainModel(modelPointCloud)
        if (ppf.savePPFTrainedModel(ppf)):
            print("Model saved correctly")
    else:
        ppf = ppf.loadPPFTrainedModel("Tjoin_ppfModelSaved_400.pkl")
        pass


    sceneName = "data/acq1_000000 - Cloud.ply"
    # sceneName = "data/acq1_ALL.ply"
    scenePointCloud = o3d.io.read_point_cloud(sceneName)
    # #####
    clustered, recomputed = ppf.matchSceneModel(scenePointCloud,1/4,False,False,True,False,3,-1)
    # #####


    # ## VISUALIZATION 
    for idxPose,pose in enumerate(recomputed):

        if idxPose>=10: break

        print(f"{pose.pose_}\n")
        transfRes = pose.pose_

        targetPCD = copy.deepcopy(modelPointCloud)
        resultPCD = copy.deepcopy(scenePointCloud)
        
        transformationResult = np.eye(4)
        transformationResult[:, :] = transfRes
        targetPCD.transform(transformationResult)
        mask = np.zeros(np.asarray(targetPCD.points).shape[0], dtype=bool)
        mask[:] = True
        colors = np.zeros((np.asarray(targetPCD.points).shape[0], 3))
        colors[mask] = [ 1.0, 0.0, 0.0]
        targetPCD.colors = o3d.utility.Vector3dVector(colors)


        # transformationResult = np.eye(4)
        # transformationResult[:, :] = resultTransformationOut
        # resultPCD.transform(transformationResult)
        mask = np.zeros(np.asarray(resultPCD.points).shape[0], dtype=bool)
        mask[:] = True
        colors = np.zeros((np.asarray(resultPCD.points).shape[0], 3))
        colors[mask] = [ 0.0, 0.0, 1.0]
        resultPCD.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([targetPCD + resultPCD])


