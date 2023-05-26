
import argparse
import tensorflow as tf
import json
import numpy as np
import os
import sys
from scipy import stats
import copy
import shutil
import open3d as o3d
import pickle as pck
import pandas as pd
import time
import glob

from Segmentation import provider
from Segmentation.utils.test_utils import *
from Segmentation.models import model
from Segmentation.utils.pc_util import write_ply_color, write_ply_normals, write_ply_scale

from Registration.PPF_registration import PPF, Pose3D 

from Segmentation.FPCC_segmentation import FPCC
from DatasetGeneration.synthethicDataGener import DataGeneration



os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
# sys.path.append(os.path.join(BASE_DIR, '../../'))
# sys.path.append(os.path.join(BASE_DIR, '../../utils'))
# sys.path.append(os.path.join(BASE_DIR, '../../models'))





class BinPicking():

    def __init__(self) -> None:
        pass


    def GenerateDataset(self, nameCadObject = "TJoin.stl", saveDataFilesBool = True):
        
        self.DataGener = DataGeneration(nameCadObject=nameCadObject, saveDataFilesBool = saveDataFilesBool)        
        self.DataGener.Generate()
      

    def FPCC_StartTrain(self, checkpointFolderName = "T_join_NoOverlap", trainingFileList = "Segmentation/data/IPAGearShaft_part_1_train.txt"):
        self.FPCC = FPCC(checkpointFolderName = checkpointFolderName)
        
        if not os.path.exists(self.FPCC.PRETRAINED_MODEL_PATH):
            os.makedirs(self.FPCC.PRETRAINED_MODEL_PATH)

        self.FPCC.LOG_STORAGE_PATH = os.path.join(self.FPCC.PRETRAINED_MODEL_PATH, 'logs')
        if not os.path.exists(self.FPCC.LOG_STORAGE_PATH):
            os.mkdir(self.FPCC.LOG_STORAGE_PATH)

        self.FPCC.SUMMARIES_FOLDER = os.path.join(self.FPCC.PRETRAINED_MODEL_PATH, 'summaries')
        if not os.path.exists(self.FPCC.SUMMARIES_FOLDER):
            os.mkdir(self.FPCC.SUMMARIES_FOLDER)

        LOG_DIR = self.FPCC.PRETRAINED_MODEL_PATH
        if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

        os.environ["CUDA_VISIBLE_DEVICES"] = self.FPCC.gpu_to_use
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        print('#### Batch Size: {0}'.format(self.FPCC.BATCH_SIZE))
        print('#### Point Number: {0}'.format(self.FPCC.POINT_NUM))
        print('#### Using GPU: {0}'.format(True if self.FPCC.gpu_to_use=='0' else False))

        self.FPCC.Train(trainingFileList=trainingFileList)


    def FPCC_StartPredict(self, checkpointFolderName = "T_join_NoOverlap", testDir = 'Segmentation/data/Tjoin/pointClouAcq1.txt', visualiseResultSegments = False):
        
        self.FPCC = FPCC(checkpointFolderName = checkpointFolderName)

        with tf.Graph().as_default():    

            if ('Segmentation/Prediction_results' in os.listdir()):
                shutil.rmtree('Segmentation/Prediction_results')
            if not os.path.exists('Segmentation/Prediction_results'):
                os.mkdir('Segmentation/Prediction_results')

            if not os.path.exists(self.FPCC.OUTPUT_DIR_PREDICT):
                os.mkdir(self.FPCC.OUTPUT_DIR_PREDICT)

            self.FPCC.OUTPUT_DIR_PREDICT_2 = os.path.join(self.FPCC.OUTPUT_DIR_PREDICT, 'scene_seg')
            if not os.path.exists(self.FPCC.OUTPUT_DIR_PREDICT_2):
                os.mkdir(self.FPCC.OUTPUT_DIR_PREDICT_2)

            self.FPCC.OUTPUT_DIR_PREDICT_3 = os.path.join(self.FPCC.OUTPUT_DIR_PREDICT, 'center_map/')
            if not os.path.exists(self.FPCC.OUTPUT_DIR_PREDICT_3):
                os.mkdir(self.FPCC.OUTPUT_DIR_PREDICT_3)

            os.environ["CUDA_VISIBLE_DEVICES"] = self.FPCC.gpu_to_use            
            os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

            print('#### Batch Size: {0}'.format(self.FPCC.BATCH_SIZE))
            print('#### Point Number: {0}'.format(self.FPCC.POINT_NUM))
            print('#### Using GPU: {0}'.format(True if self.FPCC.gpu_to_use=='0' else False))
            
            self.FPCC.Predict(testDir = testDir)

        if visualiseResultSegments:
            self.FPCC_VisualizingAfterTraining() 



    def FPCC_VisualizingAfterTraining(self):
    
        df = pd.read_csv('Segmentation/Prediction_results/PredictionResults_vdm_asm/scene_seg/_grouppred.txt', sep = ' ')
        xArr = np.array(df)

        df_c = pd.read_csv('Segmentation/Prediction_results/PredictionResults_vdm_asm/center_map/_show_center_points.txt', sep = ' ')
        xArr_c = np.array(df_c)
        xArr = np.vstack(( xArr, xArr_c))

        print("Number of centers : "+str(xArr_c.shape[0]))

        write_ply_color(xArr,'Segmentation/Prediction_results/PredictionResults_vdm_asm/PredictedSegmentsPointCloud.ply')
        pcd = o3d.io.read_point_cloud("Segmentation/Prediction_results/PredictionResults_vdm_asm/PredictedSegmentsPointCloud.ply")
        o3d.visualization.draw_geometries([pcd])


    def FPCC_VisualizingCentersAfterTraining(self):

        df = pd.read_csv('Segmentation/Prediction_results/PredictionResults_vdm_asm/scene_seg/_grouppred.txt', sep = ' ')
        xArr = np.array(df)

        df_c = pd.read_csv('Segmentation/Prediction_results/PredictionResults_vdm_asm/center_map/_show_center_points.txt', sep = ' ')
        xArr_c = np.array(df_c)

        print("Number of centers : "+str(xArr_c.shape[0]))

        #Make a buble around theese centre points
        rangeLength = 3+xArr_c.shape[0]
        diff = 0.001
        xArr_cBuble = []
        for point in xArr_c:
            point = point[:3]
            for x in range(1,rangeLength+1):
                for y in range(1,rangeLength+1):
                    for z in range(1,rangeLength+1):
                        xArr_cBuble.append([point[0]+x*diff, point[1]+y*diff, point[2]+z*diff, 0, 0, 0])
            rangeLength-=1

        xArr = np.vstack(( xArr, xArr_c))
        xArr = np.vstack(( xArr, np.array(xArr_cBuble) ))

        write_ply_color(xArr,'Segmentation/Prediction_results/PredictionResults_vdm_asm/PredictedSegmentsWithCentersPointCloud.ply')
        pcd = o3d.io.read_point_cloud("Segmentation/Prediction_results/PredictionResults_vdm_asm/PredictedSegmentsWithCentersPointCloud.ply")

        o3d.visualization.draw_geometries([pcd])


    def VisualiseColoredGroundTruthInputData(self):

        color_map = json.load(open('Segmentation/part_color_mapping.json', 'r'))
        for idxColorMap in range(len(color_map)):
            for k in range(len(color_map[idxColorMap])):
                color_map[idxColorMap][k] = round(color_map[idxColorMap][k],2)

        filePath = "Segmentation//data/IPARingScrew_part_1/train_pointcloud/"
        df = pd.read_csv(filePath+'0000_020.txt', sep = ' ')
        df.columns = ['X','Y','Z','G']

        df_out = df.iloc[:,:-1]
        segment_colors = []
        for index,row in df.iterrows():
            color = color_map[int(row[3])]
            segment_colors.append(color)

        segment_colors = np.array(segment_colors)
        df_out['R'] = segment_colors[:,0]
        df_out['G'] = segment_colors[:,1]
        df_out['B'] = segment_colors[:,2]

        df_out.to_csv(filePath+"GroundTruth.txt",encoding='utf-8', index=False, mode='w', sep=' ', header=False)



    def FPCC_PredictedSegmentsDevideIntoSeparatedFiles(self):

        df = pd.read_csv('Segmentation/Prediction_results/PredictionResults_vdm_asm/scene_seg/_grouppred.txt', sep = ' ')
        dfCoordinates = pd.read_csv('Segmentation/Prediction_results/PredictionResults_vdm_asm/_pred.txt', sep = ' ')
        df.columns = ['X','Y','Z','R','G','B']

        dft = df.groupby([df.columns[3], df.columns[4], df.columns[5]]).agg(list)

        allRowsRGB = []
        for idxR in range(dft.shape[0]):
            rowRGB = [ dft.iloc[idxR].name[0], dft.iloc[idxR].name[1], dft.iloc[idxR].name[2] ]
            allRowsRGB.append(rowRGB)

        separatedObjectPoints = []
        for r,g,b in allRowsRGB:
            points = []
            for index,row in df.iterrows():
                if row[3]==r and row[4]==g and row[5]==b:
                    # points.append(np.array(row))
                    points.append(np.hstack(((dfCoordinates.iloc[index][:-1]),np.array(row[3:]))))
            separatedObjectPoints.append(points)

        print(len(separatedObjectPoints))

        lengthsAllToSort=[]
        for lenghtObj in separatedObjectPoints:
            lengthsAllToSort.append(len(lenghtObj))

        lengthsAllToSort.sort(reverse=True)
        separatedObjectPointsSort = []

        for lengthsOfPoints in lengthsAllToSort:

            for i, point in enumerate(separatedObjectPoints):
                if len(point) == lengthsOfPoints:
                    idx = i
                    break

            separatedObjectPointsSort.append(separatedObjectPoints[idx])
            separatedObjectPoints.remove(separatedObjectPoints[idx])

        if ('PLY_Normals' in os.listdir("Segmentation/Prediction_results/PredictionResults_vdm_asm/")):
            shutil.rmtree('Segmentation/Prediction_results/PredictionResults_vdm_asm/PLY_Normals')
        os.mkdir('Segmentation/Prediction_results/PredictionResults_vdm_asm/PLY_Normals')

        if ('PLY_SegmentsOnly' in os.listdir("Segmentation/Prediction_results/PredictionResults_vdm_asm/")):
            shutil.rmtree('Segmentation/Prediction_results/PredictionResults_vdm_asm/PLY_SegmentsOnly')
        os.mkdir('Segmentation/Prediction_results/PredictionResults_vdm_asm/PLY_SegmentsOnly')
        
        for i in range(len(separatedObjectPointsSort)):
            # np.asarray(separatedObjectPointsSort[i])[:,:3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(separatedObjectPointsSort[i])[:,:3])
            pcd = pcd.voxel_down_sample(voxel_size=0.002)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=20))
            # o3d.visualization.draw_geometries([pcd],point_show_normal=True)
            write_ply_normals( pcd, f"Segmentation/Prediction_results/PredictionResults_vdm_asm/PLY_Normals/Result_Segment_{i+1}.ply", multiplyXZY=100)
            write_ply_scale( pcd, f"Segmentation/Prediction_results/PredictionResults_vdm_asm/PLY_SegmentsOnly/Result_Segment_{i+1}.ply", multiplyXZY=1)
            
        ## Saving segments with different colors
        for i in range(len(separatedObjectPointsSort)):
            write_ply_color( np.array(separatedObjectPointsSort[i]), f'Segmentation/Prediction_results/PredictionResults_vdm_asm/PLY_Color/Result_Segment_{i+1}.ply')
        print("Number of objects segmented into separated files : "+str(i+1))


    def FPCC_VisualizeOneSegmentFromPLY(self, segmendID = 0):
        pcd = o3d.io.read_point_cloud(f"Segmentation/Prediction_results/PredictionResults_vdm_asm/PLY_Color/Result_Segment_{segmendID}.ply")
        o3d.visualization.draw_geometries([pcd])


    def PPF_LoadTrainedModel(self,name) -> object:
        with open("Registration/models/"+name, 'rb') as inp:
            ppf = pck.load(inp)
        return ppf
    


    def PPF_SaveTrainedModel(self,ppfObject,modelName) -> True:
        with open(f"Registration/models/{modelName.split('/')[-1].split('.')[0]}_ppfModelSaved_{ppfObject.referenceModelPointNumber_}.pkl", 'wb') as outp:        
            pck.dump(ppfObject, outp, pck.HIGHEST_PROTOCOL)
            print(f"Saved : Registration/models/{modelName.split('/')[-1].split('.')[0]}_ppfModelSaved_{ppfObject.referenceModelPointNumber_}.pkl")
        return True



    def PPF_TrainLoadModel(self, modelName = "Registration/data/T_join.ply", loadPPFModel = 1, loadPPFModelName = "T_join_ppfModelSaved_409.pkl"):

        downSamplePointCloud = 1                    # [GearShaft=2; RingScrew=1; Tjoin=1]
        self.PPF = PPF(downSamplePointCloud,-1,100)
        self.ppf_modelPointCloud = o3d.io.read_point_cloud(modelName)

        if not loadPPFModel:
            # ppf = PPF(-1,-1,30)
            self.PPF.trainModel(self.ppf_modelPointCloud)
            if (self.PPF_SaveTrainedModel(self.PPF,modelName)):
                print("Model saved correctly")
        else:
            self.PPF = self.PPF_LoadTrainedModel(f"{loadPPFModelName}")
            pass 



    def PPF_MatchScene(self, sceneName = "Registration/data/PLY_Normals/Result_Segment_1.ply", registerAllSegments=False):

        
        if ('scaledToOriginal' in os.listdir("Registration/results/")):
            shutil.rmtree('Registration/results/scaledToOriginal')
        os.mkdir('Registration/results/scaledToOriginal')
        
        if ('scaledAsInput' in os.listdir("Registration/results/")):
            shutil.rmtree('Registration/results/scaledAsInput')
        os.mkdir('Registration/results/scaledAsInput')

        fractalOfPointsToCompareWith = 1/4
        downSamplePointCloud = 1

        if not registerAllSegments:
            # sceneName = "Registration/data/PLY_Normals/Result_Segment_1.ply"

            self.ppf_scenePointCloud = o3d.io.read_point_cloud(sceneName)

            clustered, recomputed = self.PPF.matchSceneModel(self.ppf_scenePointCloud,fractalOfPointsToCompareWith,True,downSamplePointCloud,-1)

            ## Save 
            targetPCD = copy.deepcopy(self.ppf_modelPointCloud)        
            transformationResult = np.eye(4)
            transformationResult[:, :] = recomputed[0].pose_
            targetPCD.transform(transformationResult)
            write_ply_scale(targetPCD, f"Registration/results/scaledToOriginal/Registration_Segment_{sceneName.split('.')[-2].split('_')[-1]}.ply", multiplyXZY=0.01)
            write_ply_scale(targetPCD, f"Registration/results/scaledAsInput/Registration_Segment_{sceneName.split('.')[-2].split('_')[-1]}.ply", multiplyXZY=1.0)

        else:
            for idx,fileSegment in enumerate(os.listdir("Registration/data/PLY_Normals")):
                sceneName = "Registration/data/PLY_Normals/"+fileSegment

                self.ppf_scenePointCloud = o3d.io.read_point_cloud(sceneName)

                clustered, recomputed = self.PPF.matchSceneModel(self.ppf_scenePointCloud,fractalOfPointsToCompareWith,True,downSamplePointCloud,-1)

                ## Save 
                targetPCD = copy.deepcopy(self.ppf_modelPointCloud)        
                transformationResult = np.eye(4)
                transformationResult[:, :] = recomputed[0].pose_
                targetPCD.transform(transformationResult)                
                write_ply_scale(targetPCD, f"Registration/results/scaledToOriginal/Registration_Segment_{sceneName.split('.')[-2].split('_')[-1]}.ply", multiplyXZY=0.01)
                write_ply_scale(targetPCD, f"Registration/results/scaledAsInput/Registration_Segment_{sceneName.split('.')[-2].split('_')[-1]}.ply", multiplyXZY=1.0)



        return clustered, recomputed


    def PPF_Visualization5MatchesFromLastSegment(self, poseEstimationList):

        for idxPose,pose in enumerate(poseEstimationList):

            if idxPose>=5: break

            print(f"{pose.pose_}\n")
            transfRes = pose.pose_

            targetPCD = copy.deepcopy(self.ppf_modelPointCloud)
            resultPCD = copy.deepcopy(self.ppf_scenePointCloud)
            
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



    def PPF_GetResultsFromSegmentation(self):

        if ('PLY_Normals' in os.listdir("Registration/data")):
            shutil.rmtree('Registration/data/PLY_Normals')
        os.mkdir('Registration/data/PLY_Normals')

        shutil.copytree('Segmentation/Prediction_results/PredictionResults_vdm_asm/PLY_Normals', 'Registration/data/PLY_Normals', dirs_exist_ok=True)  # Fine



########################################################################################################################


if __name__ == '__main__':

    ####################################################################################################################
    ###               It is necesarry to uncomment each line, in order to work with the algorithm                    ###
    ####################################################################################################################

    ### INIT
    binPick = BinPicking()    


    ### Data Generation
    # binPick.GenerateDataset(nameCadObject = "TJoin.stl", saveDataFilesBool = False)


    # ### TRAINING NETWORK
    # binPick.FPCC_StartTrain(checkpointFolderName = "T_join_trained_NN_", 
    #                         trainingFileList = "Segmentation/data/Generated_Dataset_Test_train.txt")


    # ### NETWORK PREDICTION 
    # binPick.FPCC_StartPredict(checkpointFolderName = "T_join_trained_NN",
    #                           testDir = 'Segmentation/data/Generated_Dataset_Test/train_pointcloud/0000_010.txt',
    #                           visualiseResultSegments = True)
    # binPick.FPCC_PredictedSegmentsDevideIntoSeparatedFiles()


    # ### Copy segmentation results to to registration
    # binPick.PPF_GetResultsFromSegmentation()


    # # ### Train / Load PPF Model
    # binPick.PPF_TrainLoadModel(modelName = "Registration/data/T_join.ply", 
    #                            loadPPFModel = True, 
    #                            loadPPFModelName = "T_join_ppfModelSaved_409.pkl")
    
    ## Model to Scene matching, requires trained model from the step above
    # clustered, recomputed = binPick.PPF_MatchScene(sceneName = "Registration/data/PLY_Normals/Result_Segment_1.ply",
    #                                                registerAllSegments=False)

    # binPick.PPF_Visualization5MatchesFromLastSegment(recomputed)












