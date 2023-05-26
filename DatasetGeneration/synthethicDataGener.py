
# import tensorflow as tf
import numpy as np
import os
import pandas as pd
import open3d as o3d
import json
import shutil
import copy
import math
import time
from PIL import Image



class DataGeneration():

    def __init__(self, nameCadObject = "TJoin.stl", saveDataFilesBool = True) -> None:

        ## IMPORT COLOURS
        self.color_map = json.load(open('DatasetGeneration/data/part_color_mapping.json', 'r'))
        for idxColorMap in range(len(self.color_map)):
            for k in range(len(self.color_map[idxColorMap])):
                self.color_map[idxColorMap][k] = round(self.color_map[idxColorMap][k],2)

        ## SET THE VALUES 

        # Save Data
        self.saveDataFiles = saveDataFilesBool
        self.overLapingPrevent = True

        ## Setting the free enviroment
        self.rootFolder = 'DatasetGeneration/data/Generated_Dataset_Test/'
        self.typeFolderGT = 'gt/'
        self.typeFolderTrain = 'train_pointcloud/'
        self.typeFolderRGB = 'rgb/'

        # 3D CAD OBJECT 
        self.nameObject = nameCadObject
        self.meshCAD = o3d.io.read_triangle_mesh(f'DatasetGeneration/data/{self.nameObject}')

        # Number of cycles - scenes
        self.cycleMaxRange = 50

        #Number of maximal objects
        self.objectsMaxNumber = 10
        self.maxOverlapObjectsRandGener = 1000

        self.numberOfPointsForObjectSample = 8000




    def Generate(self):

        if not self.saveDataFiles:
            self.cycleMaxRange = 1

        ## ===================================================================
        ## ===================================================================

        if self.saveDataFiles:
            if( self.rootFolder.split("/")[-2] not in os.listdir('/'.join(self.rootFolder.split("/")[:-2])) ):
                os.mkdir(self.rootFolder)

            if( 'gt' in os.listdir(self.rootFolder)): shutil.rmtree(self.rootFolder+self.typeFolderGT)
            os.mkdir(self.rootFolder+self.typeFolderGT)

            if( 'train_pointcloud' in os.listdir(self.rootFolder)): shutil.rmtree(self.rootFolder+self.typeFolderTrain)
            os.mkdir(self.rootFolder+self.typeFolderTrain)

            if( 'rgb' in os.listdir(self.rootFolder)): shutil.rmtree(self.rootFolder+self.typeFolderRGB)
            os.mkdir(self.rootFolder+self.typeFolderRGB)


        # Create the visualization window
        vis = o3d.visualization.Visualizer()
        if self.saveDataFiles : vis.create_window(visible=False)

        #############################################################################################
        #############################################################################################

        for cycle in range(self.cycleMaxRange):

            if self.saveDataFiles:
                cycleFolderName = "cycle_%04d"%(cycle)
                if( cycleFolderName in os.listdir(self.rootFolder+self.typeFolderGT)): shutil.rmtree(self.rootFolder+self.typeFolderGT+cycleFolderName)
                os.mkdir(self.rootFolder+self.typeFolderGT+"cycle_%04d"%(cycle))

                if( cycleFolderName in os.listdir(self.rootFolder+self.typeFolderRGB)): shutil.rmtree(self.rootFolder+self.typeFolderRGB+cycleFolderName)
                os.mkdir(self.rootFolder+self.typeFolderRGB+"cycle_%04d"%(cycle))

                # Create a sample DataFrame
                df = pd.DataFrame({'id': [0],'class': [0],'x': [0],'y': [0],'z': [0],'rot_x_axis_1': [0],'rot_x_axis_2': [0],'rot_x_axis_3': [0],'rot_y_axis_1': [0],'rot_y_axis_2': [0],'rot_y_axis_3': [0],'rot_z_axis_1': [0],'rot_z_axis_2': [0],'rot_z_axis_3': [0],'visibility_score_o': [0],'visibility_score_p': [0]})
                header_df_center = pd.DataFrame(columns=df.columns)

            differentColorsGroups = []

            print(f"Cycle {cycle} start : "+time.strftime("%H:%M:%S", time.localtime()))

            if self.overLapingPrevent:
                X_overLaping = [[-10000,-10000]]
                Y_overLaping = [[-10000,-10000]]
                Z_overLaping = [[-10000,-10000]]
                # X_overLaping = np.asarray([-10000,-10000])
                # Y_overLaping = np.asarray([-10000,-10000])
                # Z_overLaping = np.asarray([-10000,-10000])


            #############################################################################################
            ## FIRST - ALL BLACK OBJECT
            #############################################################################################
            # transformation = np.eye(4)
            # point_cloud = self.meshCAD.sample_points_uniformly(number_of_points=self.numberOfPointsForObjectSample)
            # num_points = len(point_cloud.points)
            # mask = np.zeros(num_points, dtype=bool)
            # mask[:] = True
            # colors = np.zeros((num_points, 3))
            # colors[mask] = [ 0.0, 0.0, 0.0]
            # differentColorsGroups = [[0,0,0]]
            # point_cloud.colors = o3d.utility.Vector3dVector(colors)
            # # vis.add_geometry(point_cloud)
            # pcd_combined = point_cloud
            # # Making .csv file with with their appropriate centers
            # if self.saveDataFiles:
                # point_cloudCenter = [ np.sum(np.asarray(point_cloud.points)[:,0]) / num_points , np.sum(np.asarray(point_cloud.points)[:,1]) / num_points , np.sum(np.asarray(point_cloud.points)[:,2]) / num_points ] 
                # addRow = np.zeros((header_df_center.shape[1]))
                # addRow[2:5] = point_cloudCenter
                # header_df_center.loc[len(header_df_center.index)] = addRow
                # header_df_center.to_csv(self.rootFolder+self.typeFolderGT+cycleFolderName+"/%03d.csv"%(1), encoding='utf-8', index=False)

            # pcd = pcd_combined
            # diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
            # # o3d.visualization.draw_geometries([pcd])
            # camera = [0, 0, diameter]
            # radius = diameter * 4000
            # _, pt_map = pcd.hidden_point_removal(camera, radius)
            # pcd = pcd.select_by_index(pt_map)
            # # o3d.visualization.draw_geometries([pcd])
            # if self.saveDataFiles:
                # xyz = np.asarray(pcd.points)
                # rgb = np.asarray(pcd.colors)
                # point_cloud_np = np.hstack((xyz,rgb))

                # # Spliting into groups according their colours
                # arrayOfGroups = []
                # colorGroupIdx = 0
                # for searchedColor in differentColorsGroups:
                #     for idxColorGroup,rgbA in enumerate(rgb): 
                #         if(rgbA[0]==searchedColor[0] and rgbA[1]==searchedColor[1] and rgbA[2]==searchedColor[2]): arrayOfGroups.append([colorGroupIdx])
                #     # print(len(arrayOfGroups))
                #     colorGroupIdx += 1

                # xyzGID = np.hstack((xyz, np.asarray(arrayOfGroups)))

                # transform = pd.DataFrame({'x': [0],'y': [0],'z': [0],'gID': [0]})
                # header_transform = pd.DataFrame(columns=df.columns)

                # df_train = pd.DataFrame({'x': [0],'y': [0],'z': [0],'gID': [0]})
                # header_train = pd.DataFrame(columns=df_train.columns)
                # pc_trainDF = pd.DataFrame(xyzGID, columns=header_train.columns)
                # pc_trainDF['x'] = pc_trainDF['x'].round(6)
                # pc_trainDF['y'] = pc_trainDF['y'].round(6)
                # pc_trainDF['z'] = pc_trainDF['z'].round(6)
                # pc_trainDF['gID'] = pc_trainDF['gID'].astype(int)
                # pc_trainDF.to_csv(self.rootFolder+self.typeFolderTrain+"/%04d_%03d.txt"%(cycle,1), encoding='utf-8', index=False, mode='w', sep=' ', header=False)

                # vis.add_geometry(pcd)
                # vis.update_renderer()   
                # image = vis.capture_screen_float_buffer(do_render=True)
                # image_pil = Image.fromarray((np.asarray(image)* 255).astype('uint8'))
                # image_pil.save(self.rootFolder+self.typeFolderRGB+cycleFolderName+"/%03d.bmp"%(1))
                # print(self.rootFolder.split('.')[1]+"**/cycle_%04d/%03d"%(cycle,1)+"   : %d points"%(len(xyz)) )
                # vis.remove_geometry(pcd)
            #############################################################################################
            #############################################################################################




            for idxObject in range(self.objectsMaxNumber):
                # Load the second CAD model
                self.meshCAD_Orig = o3d.io.read_triangle_mesh(f"DatasetGeneration/data/{self.nameObject}")
                self.meshCAD = copy.deepcopy(self.meshCAD_Orig)
                point_cloud = self.meshCAD.sample_points_uniformly(number_of_points=self.numberOfPointsForObjectSample)

                maxCadCenterHeight = abs(np.max(np.asarray(point_cloud.points)[:,2]))+abs(np.min(np.asarray(point_cloud.points)[:,2])) * 0.25
                # maxCadHeightCenterLambda = 0.7 * 0.25 * 0.5

                resetObject = True
                counterIdx = -1
                rotateObj = True

                while resetObject:

                    # rotateObj = not rotateObj
                    counterIdx+=1
                    self.meshCAD = copy.deepcopy(self.meshCAD_Orig)
                    point_cloud = self.meshCAD.sample_points_uniformly(number_of_points=500)
                    # point_cloud1 = copy.deepcopy(point_cloud)
                    resetObject = False
                    # Place the second CAD model at a random position and orientation     

                    transformation1 = np.eye(4)
                    transformation1[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz( (float(np.random.normal(-1,1,1)*np.pi*0.5 + 0.5*np.pi),0,0) )
                    # transformation1[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz( (float(np.random.normal( 0.5*np.pi,.25,1)), 0, 0) )
                    # else:
                    #     transformation1[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz(( float(((np.random.rand(1)*0.25)+0.375)*np.pi), 0, 0 )) #np.random.rand(0,1,1)

                    # transformation1[:2, 3] = np.random.rand(2) * 0.15
                    transformation1[:2, 3] = np.random.uniform(-1,1,2) * 0.15

                    # transformation1[2, 3] = np.random.rand(1) * maxCadCenterHeight * 0.5
                    transformation1[2, 3] = np.random.normal(0,1,1) * maxCadCenterHeight * 0.4
                    # transformation1[2, 3] = np.random.randint(0,2) * maxCadCenterHeight * 0.5
                    point_cloud.transform(transformation1)

                    transformation2 = np.eye(4)
                    transformation2[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz(( 0, 0, float(np.random.normal( 0,2*np.pi,1)) ))
                    # transformation2[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz(( 0, 0, float(np.random.uniform(0,1,1)*2*np.pi) ))
                    point_cloud.transform(transformation2)
                    # point_cloud3 = copy.deepcopy(point_cloud)
                    
                    # o3d.visualization.draw_geometries([point_cloud3+point_cloud1+point_cloud2])
                    # vis.remove_geometry([point_cloud3+point_cloud1+point_cloud2])
                    bestTransformPose = (transformation1, transformation2)


                    if self.overLapingPrevent:

                        pcPoints = np.asarray(point_cloud.points)
                        countPointsOverlappingActual = 0

                        for idxObj,_ in enumerate(X_overLaping):
                            # if (resetObject): break
                            if(counterIdx > self.maxOverlapObjectsRandGener): 
                                # print("!!",end="")
                                break

                            for (x,y,z) in pcPoints:
                                result1,result2,result3 = False,False,False
                                # if (resetObject): break

                                # minimum comparing X
                                if (x) > (X_overLaping[idxObj][0])  and (x) < (X_overLaping[idxObj][1]):
                                    result1=True
                                # minimum comparing Y
                                if (y) > (Y_overLaping[idxObj][0])  and (y) < (Y_overLaping[idxObj][1]):
                                    result2=True
                                # minimum comparing Z
                                if (z) > (Z_overLaping[idxObj][0])  and (z) < (Z_overLaping[idxObj][1]):
                                    result3=True

                                if( result1 and result2 and result3 ):
                                    resetObject=True
                                    countPointsOverlappingActual += 1
                                    # break                                            
                            
                        ## Decide the best possition if all were overlapping
                        if counterIdx == 0:
                            bestTransformPose = (transformation1, transformation2)
                            counterPoitsOverlapBest = countPointsOverlappingActual
                            
                        if( countPointsOverlappingActual < counterPoitsOverlapBest and counterIdx < self.maxOverlapObjectsRandGener ):
                            bestTransformPose = (transformation1, transformation2)
                            counterPoitsOverlapBest = countPointsOverlappingActual

                                
                        if not resetObject:                    
                            self.meshCAD = copy.deepcopy(self.meshCAD_Orig)
                            point_cloud = self.meshCAD.sample_points_uniformly(number_of_points=300)
                            transformation1, transformation2 = bestTransformPose            
                            point_cloud.transform(transformation1)
                            point_cloud.transform(transformation2)
                            X_overLaping.append([np.min(np.asarray(point_cloud.points)[:,0]), np.max(np.asarray(point_cloud.points)[:,0])])
                            Y_overLaping.append([np.min(np.asarray(point_cloud.points)[:,1]), np.max(np.asarray(point_cloud.points)[:,1])])
                            Z_overLaping.append([np.min(np.asarray(point_cloud.points)[:,2]), np.max(np.asarray(point_cloud.points)[:,2])])
                    

                transformation1, transformation2 = bestTransformPose
                if self.overLapingPrevent:
                    print(f"Overlapped pts: {counterPoitsOverlapBest}; ",end="")

                self.meshCAD = copy.deepcopy(self.meshCAD_Orig)
                self.meshCAD.transform(transformation1)
                self.meshCAD.transform(transformation2)
                point_cloud = self.meshCAD.sample_points_uniformly(number_of_points=self.numberOfPointsForObjectSample)

                num_points = len(point_cloud.points)
                mask = np.zeros(num_points, dtype=bool)
                mask[:] = True
                colors = np.zeros((num_points, 3))
                colors[mask] = self.color_map[idxObject]
                differentColorsGroups.append(self.color_map[idxObject])
                point_cloud.colors = o3d.utility.Vector3dVector(colors)

                if self.saveDataFiles:
                    # Making .csv file with with their appropriate centers
                    point_cloudCenter = [ np.sum(np.asarray(point_cloud.points)[:,0]) / num_points , np.sum(np.asarray(point_cloud.points)[:,1]) / num_points , np.sum(np.asarray(point_cloud.points)[:,2]) / num_points ] 
                    addRow = np.zeros((header_df_center.shape[1]))
                    addRow[2:5] = point_cloudCenter
                    header_df_center.loc[len(header_df_center.index)] = addRow
                    header_df_center.to_csv(self.rootFolder+self.typeFolderGT+cycleFolderName+"/%03d.csv"%(idxObject+1), encoding='utf-8', index=False, mode='w')

                if idxObject==0: pcd_combined = point_cloud
                else: pcd_combined += point_cloud

                pcd = pcd_combined
                diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
                # o3d.visualization.draw_geometries([pcd])
                camera = [0, 0, diameter]
                radius = diameter * 4000
                _, pt_map = pcd.hidden_point_removal(camera, radius)
                pcd = pcd.select_by_index(pt_map)

                if not self.saveDataFiles:
                    if idxObject==0: pcd_view_combined = pcd
                    else: pcd_view_combined += pcd
                # o3d.visualization.draw_geometries([pcd])
                xyz = np.asarray(pcd.points)

                if self.saveDataFiles:
                    xyz = np.asarray(pcd.points)
                    rgb = np.asarray(pcd.colors)
                    point_cloud_np = np.hstack((xyz,rgb))

                    # Spliting into groups according their colours
                    arrayOfGroups = []
                    colorGroupIdx = 0
                    for searchedColor in differentColorsGroups:
                        for idxColorGroup,rgbA in enumerate(rgb): 
                            if(rgbA[0]==searchedColor[0] and rgbA[1]==searchedColor[1] and rgbA[2]==searchedColor[2]): arrayOfGroups.append([colorGroupIdx])
                        # print(len(arrayOfGroups))
                        colorGroupIdx += 1

                    xyzGID = np.hstack((xyz, np.asarray(arrayOfGroups)))

                    transform = pd.DataFrame({'x': [0],'y': [0],'z': [0],'gID': [0]})
                    header_transform = pd.DataFrame(columns=df.columns)

                    df_train = pd.DataFrame({'x': [0],'y': [0],'z': [0],'gID': [0]})
                    header_train = pd.DataFrame(columns=df_train.columns)
                    pc_trainDF = pd.DataFrame(xyzGID, columns=header_train.columns)
                    pc_trainDF['x'] = pc_trainDF['x'].round(6)
                    pc_trainDF['y'] = pc_trainDF['y'].round(6)
                    pc_trainDF['z'] = pc_trainDF['z'].round(6)
                    pc_trainDF['gID'] = pc_trainDF['gID'].astype(int)
                    pc_trainDF.to_csv(self.rootFolder+self.typeFolderTrain+"/%04d_%03d.txt"%(cycle,idxObject+1), encoding='utf-8', index=False, mode='w', sep=' ', header=False)

                    vis.add_geometry(pcd)
                    vis.update_renderer()   
                    image = vis.capture_screen_float_buffer(do_render=True)
                    image_pil = Image.fromarray((np.asarray(image)* 255).astype('uint8'))
                    image_pil.save(self.rootFolder+self.typeFolderRGB+cycleFolderName+"/%03d.bmp"%(idxObject+1))
                    vis.remove_geometry(pcd)

                print(self.rootFolder+"**/cycle_%04d/%03d"%(cycle,idxObject+1)+"  :  %d points"%(len(xyz)) )

                # if self.overLapingPrevent:
                #     X_overLaping = np.asarray(X_overLaping)
                #     Y_overLaping = np.asarray(Y_overLaping)
                #     Z_overLaping = np.asarray(Z_overLaping)

        if not self.saveDataFiles:
            o3d.visualization.draw_geometries([pcd_view_combined])

