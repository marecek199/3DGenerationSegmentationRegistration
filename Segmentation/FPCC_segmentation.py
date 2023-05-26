
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
from utils.pc_util import write_ply_color, write_ply_normals
import time
import provider
from utils.test_utils import *
from models import model
import glob



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../../'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../models'))




class FPCC():

    def __init__(self, checkpointFolderName = "T_join_NoOverlap") -> None:
        
        ## Restore JSON Test/Training variables
        with open("Segmentation/FPCC_VariablesData.json") as json_file:
            self.dataVariables = json.load(json_file)



        #####           SETTINGS            ####
        self.gpu_to_use = self.dataVariables["gpu"]
        self.BACKBONE = self.dataVariables["backbone"] 

        self.PRETRAINED_MODEL_PATH = os.path.join(self.dataVariables["restore_model_dir"],f'{checkpointFolderName}/')

        ## TRAIN 
        # self.TRAINING_FILE_LIST = "Segmentation/data/IPAGearShaft_part_1_train.txt"
        self.printLastEpochStats = True
        self.TRAINING_EPOCHES = self.dataVariables["epoch"]
        self.POINT_DIM = self.dataVariables["point_dim"]
        self.POINT_NUM = self.dataVariables["point_num"]
        self.BATCH_SIZE = self.dataVariables["batch"]
        self.vdm = self.dataVariables["use_vdm"]
        self.asm = self.dataVariables["use_asm"]
        self.MARGINS = [self.dataVariables["margin_same"], self.dataVariables["margin_diff"]]
        self.DECAY_STEP = 800000.
        self.DECAY_RATE = 0.5
        self.LEARNING_RATE_CLIP = 1e-6
        self.BASE_LEARNING_RATE = 1e-4
        self.R_maxTrain = self.dataVariables["R_max_Train"]
        self.NUM_GROUPS = self.dataVariables["group_num"]



        ## TRAIN
        # self.TEST_DIR = 'Segmentation/data/Tjoin/pointClouAcq1.txt'
        self.OUTPUT_VERBOSE = True
        self.R_maxTest = self.dataVariables["R_max_Test"]
        self.OUTPUT_DIR_PREDICT = os.path.join('Segmentation/Prediction_results/PredictionResults_vdm_asm/')
        self.SAMPLE_LIMIT = self.BATCH_SIZE
        self.center_score_th = self.dataVariables["center_score_th"]
        self.max_feature_distance = None # The point whose feature distance from the center point is greater than this value is regarded as noise [-1]
        self.max_3d_distance = 1. # The farthest distance from the point to the center. usually max_3d_distance > r_nms





    def Printout(self, flog, data):
        print(data)
        flog.write(data + '\n')

    def Samples(self, data, sample_num_point,limit=None):

        N = data.shape[0]
        dim =  data.shape[-1]
        order = np.arange(N)
        np.random.shuffle(order)

        data = data[order, :]

        if limit == None:
            batch_num = int(np.ceil(N / float(sample_num_point)))
        else:
            batch_num = min(int(np.ceil(N / float(sample_num_point))),limit)

        sample_datas = np.zeros((batch_num, sample_num_point, dim))


        for i in range(batch_num):
            beg_idx = i*sample_num_point
            end_idx = min((i+1)*sample_num_point, N)
            num = end_idx - beg_idx
            sample_datas[i,0:num,:] = data[beg_idx:end_idx, :]

            if num < sample_num_point:
                # print('makeup')
                makeup_indices = np.random.choice(N, sample_num_point - num)
                sample_datas[i,num:,:] = data[makeup_indices, :]

        return sample_datas

    def Samples_reshape_txt(self, data_label, num_point=4096,limit=None):
        """ input: [X,Y,Z]  shape：（N,3）or [X,Y,Z, inslab] (N,4)
            for XYZ, add normalized XYZ as 678 channels and aligned XYZ as 345 channels

            return:
            x,y,z,x0,y0,z0, Nx,Ny,Nz
        """
        dim = data_label.shape[-1]
        xyz_min = np.amin(data_label, axis=0)[0:3]
        xyz_max = np.amax(data_label, axis=0)[0:3]

        data_align = np.zeros((data_label.shape[0], dim+6))

        xyz_min = np.amin(data_label, axis=0)[0:3]
        data_align[:,0:3] = data_label[:,0:3]
        data_align[:,3:6] = data_label[:,0:3]-xyz_min
        data_align[:,-1] = data_label[:,-1]
        data_align[:,-2] = data_label[:,-2]

        max_x = max(data_align[:,3])
        max_y = max(data_align[:,4])
        max_z = max(data_align[:,5])

        data_batch  = self.Samples(data_align, num_point,limit)
        batch_num = data_batch.shape[0]


        new_data_batch = np.zeros((batch_num, num_point, 9))

        for b in range(batch_num):
            new_data_batch[b, :, 6] = data_batch[b, :, 3]/max_x
            new_data_batch[b, :, 7] = data_batch[b, :, 4]/max_y
            new_data_batch[b, :, 8] = data_batch[b, :, 5]/max_z

        new_data_batch[:, :, 0:6] = data_batch[:,:,0:6]
        gt =  data_batch[:,:,-1]

        return new_data_batch, gt






    def Predict(self, testDir = 'Segmentation/data/Tjoin/pointClouAcq1.txt'):


        self.TEST_DIR = testDir

        is_training = False

        with tf.device('/gpu:' + str(self.gpu_to_use)):
            is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=())

            pointclouds_ph, _, _ = \
                model.placeholder_inputs(self.BATCH_SIZE, self.POINT_NUM, 50, self.POINT_DIM)

            net_output = model.get_model( self.BACKBONE, pointclouds_ph,is_training_ph,train=False)

        # Add ops to save and restore all the variables.

        saver = tf.compat.v1.train.Saver()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        foundTestingDirectory = False

        with tf.compat.v1.Session(config=config) as sess:

            flog = open(os.path.join(self.OUTPUT_DIR_PREDICT, 'log.txt'), 'w')

            # Restore variables from disk.

            ckptstate = tf.train.get_checkpoint_state(self.PRETRAINED_MODEL_PATH)
            if ckptstate is not None:
                LOAD_MODEL_FILE = os.path.join(self.PRETRAINED_MODEL_PATH,os.path.basename(ckptstate.model_checkpoint_path))
                saver.restore(sess, LOAD_MODEL_FILE)
                self.Printout(flog, "Model loaded in file: %s" % LOAD_MODEL_FILE)
            else:
                self.Printout(flog, "Fail to load modelfile: %s" % self.PRETRAINED_MODEL_PATH)


            un_gt_list = []
            output_filelist_f = os.path.join(self.OUTPUT_DIR_PREDICT, 'output_filelist.txt')
            fout_out_filelist = open(output_filelist_f, 'w')
            for f in glob.glob(self.TEST_DIR):

                foundTestingDirectory=True
                file_name = ""
                # file_name = f.split("\\")[-1].split('.')[0].split('/')[-1]
                print(file_name)
                if not os.path.exists(f):
                    print('%s is not exists',f)
                    continue
                # scene_start = time.time()
                points = np.loadtxt(f)[:,:] # points: [XYZ,instance label]
                points_num = points.shape[0]

                input_data, gt_batch = self.Samples_reshape_txt(points,num_point=self.POINT_NUM,limit=self.SAMPLE_LIMIT)
                # input_data： [original XYZ, aligned XYZ, normalizated XYZ] N x 9 (without RGB)
                
                out_data_label_filename = file_name + '_pred.txt'
                out_data_label_filename = os.path.join(self.OUTPUT_DIR_PREDICT, out_data_label_filename)
                out_gt_label_filename = file_name + '_gt.txt'
                out_gt_label_filename = os.path.join(self.OUTPUT_DIR_PREDICT, out_gt_label_filename)
                fout_data_label = open(out_data_label_filename, 'w')
                fout_gt_label = open(out_gt_label_filename, 'w')
                fout_out_filelist.write(out_data_label_filename+'\n')



                valid_batch_num = input_data.shape[0]
                predict_num = int(np.ceil(valid_batch_num / self.BATCH_SIZE))
                point_features = []
                pts_scores = []

                for n in range(predict_num):
                    feed_data = np.zeros((self.BATCH_SIZE, self.POINT_NUM, self.POINT_DIM))
                    beg_idx = n * self.BATCH_SIZE
                    end_idx = min((n+1)*self.BATCH_SIZE, valid_batch_num)
                    num = end_idx - beg_idx
                    feed_data[:num,:,:] = input_data[beg_idx:end_idx,:,3:3+self.POINT_DIM]

                    feed_dict = {
                    pointclouds_ph: feed_data,
                    is_training_ph: is_training,
                    }

                    point_feature, pts_score_val0= \
                    sess.run([net_output['point_features'],
                    net_output['center_score']],
                    feed_dict=feed_dict)

                    point_features.append([point_feature])
                    pts_scores.append([pts_score_val0])

                pred_score_val = np.concatenate(pts_scores,axis=0)
                point_features = np.concatenate(point_features,axis=0)

                
                input_data = input_data.reshape([-1, 3+self.POINT_DIM])
                input_data = input_data[:points_num,:]

                pred_score_val = pred_score_val.reshape([-1,1])
                pred_score_val = pred_score_val[:points_num]

                point_features = point_features.reshape([-1,128])
                point_features = point_features[:points_num,:]

                group_pred, c_index = GroupMerging_fpcc(input_data[:,3:6],point_features, pred_score_val, \
                    center_socre_th = self.center_score_th, max_feature_dis=self.max_feature_distance, use_3d_mask=self.max_3d_distance, r_nms=self.R_maxTest)
                # scene_end = scene_start- time.timeself.()

                # c_score = pred_score_val[c_index]
                pts = input_data

                ###### Generate Results for Evaluation
                group_pred_final = group_pred.reshape(-1)
                group_gt = gt_batch.reshape(-1)
                # seg_pred = np.zeros((group_pred.shape))

                ins_pre = group_pred_final.astype(np.int32)

                ins_gt = group_gt
                # un_gt = np.unique(ins_gt)
                # un_gt_list.append(len(un_gt))
                for i in range(pts.shape[0]):
                    fout_data_label.write('%f %f %f %d\n' % (pts[i, 0], pts[i, 1], pts[i, 2], ins_pre[i]))
                    fout_gt_label.write('%d\n' % ins_gt[i])

                fout_data_label.close()
                fout_gt_label.close()
                
                
                if self.OUTPUT_VERBOSE:
                    output_color_point_cloud(pts[:, 3:6], ins_pre.astype(np.int32),
                        os.path.join(self.OUTPUT_DIR_PREDICT_2, '%s_grouppred.txt' % (file_name)))
                    
                    output_color_point_cloud_centers(pts[:, 3:6], ins_pre.astype(np.int32), c_index, 
                        os.path.join(self.OUTPUT_DIR_PREDICT_3, '%s_show_center_points.txt' % (file_name)))

                    output_color_point_center_score(pts[:, 3:6], pred_score_val,os.path.join(self.OUTPUT_DIR_PREDICT_3,'%s_c_map.txt' % (file_name)))

            if not foundTestingDirectory: print("Testing file not founded ...\nShowing results of the last prediction")
            fout_out_filelist.close()



    def Train(self, trainingFileList = "Segmentation/data/IPAGearShaft_part_1_train.txt"):

        self.TRAINING_FILE_LIST = trainingFileList
        with tf.Graph().as_default():

            ## setting for GPU training
            with tf.device('/gpu:' + str(self.gpu_to_use)):
                batch = tf.Variable(0, trainable=False, name='batch')

                learning_rate = tf.compat.v1.train.exponential_decay(
                    self.BASE_LEARNING_RATE,  # base learning rate
                    batch * self.BATCH_SIZE,  # global_var indicating the number of steps
                    self.DECAY_STEP,  # step size
                    self.DECAY_RATE,  # decay rate
                    staircase=True  # Stair-case or continuous decreasing
                )

                learning_rate = tf.maximum(learning_rate, self.LEARNING_RATE_CLIP)

                lr_op = tf.summary.scalar('learning_rate', learning_rate)

                pointclouds_ph, ptsgroup_label_ph, pts_score_ph = \
                    model.placeholder_inputs(self.BATCH_SIZE, self.POINT_NUM, self.NUM_GROUPS, self.POINT_DIM)
                is_training_ph = tf.compat.v1.placeholder(tf.bool, shape=())

                labels = {'ptsgroup': ptsgroup_label_ph,
                        # 'semseg': ptsseglabel_ph,
                        # 'semseg_mask': pts_seglabel_mask_ph,
                        # 'group_mask': pts_group_mask_ph,
                        'center_score': pts_score_ph}

                net_output = model.get_model(self.BACKBONE, pointclouds_ph, is_training_ph)
                loss, score_loss, grouperr = model.get_loss(net_output, labels, self.vdm, self.asm, self.R_maxTrain, self.MARGINS)

                total_training_loss_ph = tf.compat.v1.placeholder(tf.float32, shape=())
                group_err_loss_ph = tf.compat.v1.placeholder(tf.float32, shape=())
                total_train_loss_sum_op = tf.summary.scalar('total_training_loss', total_training_loss_ph)
                group_err_op = tf.summary.scalar('group_err_loss', group_err_loss_ph)


            train_variables = tf.compat.v1.trainable_variables()

            ## Using optimizer ADAM
            trainer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            train_op = trainer.minimize(loss, var_list=train_variables, global_step=batch)

            loader = tf.compat.v1.train.Saver([v for v in tf.compat.v1.all_variables()#])
                                    if
                                    ('conf_logits' not in v.name) and
                                        ('Fsim' not in v.name) and
                                        ('Fsconf' not in v.name) and
                                        ('batch' not in v.name)
                                    ])

            saver = tf.compat.v1.train.Saver([v for v in tf.compat.v1.all_variables()])

            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            # config.gpu_options.visible_device_list = '0'
            config.allow_soft_placement = True
            sess = tf.compat.v1.Session(config=config)

            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)


            train_writer = tf.compat.v1.summary.FileWriter(self.SUMMARIES_FOLDER + '/train', sess.graph)      


            train_file_list = provider.getDataFiles(str("./"+str(self.TRAINING_FILE_LIST)))
            num_train_file = len(train_file_list)

            fcmd = open(os.path.join(self.LOG_STORAGE_PATH, 'cmd.txt'), 'w')
            fcmd.write(str(self.dataVariables))
            fcmd.close()

            log_file = time.strftime('log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            flog = open(os.path.join(self.LOG_STORAGE_PATH, log_file + '.txt'), 'w')
            self.epochContinueNumber = 0

            ckptstate = tf.train.get_checkpoint_state(self.PRETRAINED_MODEL_PATH)
            if ckptstate is not None:
                LOAD_MODEL_FILE = os.path.join(self.PRETRAINED_MODEL_PATH, os.path.basename(ckptstate.model_checkpoint_path))
                loader.restore(sess, LOAD_MODEL_FILE)
                self.epochContinueNumber = int(LOAD_MODEL_FILE.split('_')[-1].split('.')[0])
                self.Printout(flog, "Model loaded in file: %s" % LOAD_MODEL_FILE)
            else:
                self.Printout(flog, "Fail to load modelfile: %s" % self.PRETRAINED_MODEL_PATH)


            train_file_idx = np.arange(0, len(train_file_list))
            np.random.shuffle(train_file_idx)

            ## load all data into memory
            all_data = []
            all_group = []
            all_seg = []
            all_score = []
            for i in range(num_train_file):
                cur_train_filename = train_file_list[train_file_idx[i]]
                self.Printout(flog, 'Loading train file ' + cur_train_filename +'\t'+ str(i)+'/'+str(num_train_file))
                cur_data, cur_group, _, _, cur_score = provider.loadDataFile_with_groupseglabel_stanfordindoor(cur_train_filename)
                # cur_data = cur_data.reshape([-1,4096,3])

                all_data += [cur_data]
                all_group += [cur_group]
                all_score += [cur_score]

            all_data = np.concatenate(all_data,axis=0)
            all_group = np.concatenate(all_group,axis=0)
            all_score = np.concatenate(all_score,axis=0)


            num_data = all_data.shape[0]
            num_batch = num_data // self.BATCH_SIZE



            def train_one_epoch(epoch_num):

                ### NOTE: is_training = False: 
                ### do not update bn parameters during training due to the small batch size. This requires pre-training PointNet with large batchsize (say 32).
                is_training = False

                order = np.arange(num_data)
                np.random.shuffle(order)

                total_loss = 0.0
                total_score_loss = 0.0
                total_grouperr = 0.0
                
                num_batch_print_echo = num_batch // 10

                for j in range(num_batch):
                    # if(j % num_batch_print_echo == 0):                
                    # print(".",end="")

                    begidx = j * self.BATCH_SIZE
                    endidx = (j + 1) * self.BATCH_SIZE

                    # pts_label_one_hot, pts_label_mask = model.convert_seg_to_one_hot(all_seg[order[begidx: endidx]])
                    pts_group_label, _ = model.convert_groupandcate_to_one_hot(all_group[order[begidx: endidx]], NUM_GROUPS=self.NUM_GROUPS)                
                    pts_score = all_score[order[begidx: endidx]]
                    input_data = all_data[order[begidx: endidx], ...]

                    feed_dict = {
                        pointclouds_ph: input_data[...,:self.POINT_DIM],
                        ptsgroup_label_ph: pts_group_label,
                        pts_score_ph:pts_score,
                        is_training_ph: is_training,
                    }

                    _, loss_val,score_loss_val, grouperr_val = sess.run([train_op, loss, score_loss, grouperr], feed_dict=feed_dict)

                    total_loss += loss_val
                    total_score_loss += score_loss_val


                    total_grouperr += grouperr_val


                    if j % num_batch_print_echo == num_batch_print_echo-1 or self.printLastEpochStats:
                        # print()
                        if self.printLastEpochStats : print('Last Epoch Stats :')
                        self.Printout(flog, time.strftime("%H:%M:%S", time.localtime()) + ' - Batch: %d / %d, loss: %f, score_loss: %f, grouperr: %f' % (j, num_batch, total_loss/100, total_score_loss/100, total_grouperr/100))
                        # lr_sum, batch_sum, train_loss_sum, group_err_sum = sess.run( [lr_op, batch, total_train_loss_sum_op, group_err_op], feed_dict={total_training_loss_ph: total_loss/100 , group_err_loss_ph: total_grouperr / 100 })
                        # # lr_sum, batch_sum, train_loss_sum, group_err_sum = sess.run( \
                        # #     [lr_op, batch, total_train_loss_sum_op, group_err_op], \
                        # #     feed_dict={total_training_loss_ph: total_loss / 100.,
                        # #                group_err_loss_ph: total_grouperr / 100., })

                        # train_writer.add_summary(train_loss_sum, batch_sum)
                        # train_writer.add_summary(lr_sum, batch_sum)
                        # train_writer.add_summary(group_err_sum, batch_sum)
                        self.printLastEpochStats = False

                        total_grouperr = 0.0
                        total_loss = 0.0
                        total_score_loss = 0.0


            if not os.path.exists(self.PRETRAINED_MODEL_PATH):
                os.mkdir(self.PRETRAINED_MODEL_PATH)

            
            for epoch in range(self.TRAINING_EPOCHES):
                self.Printout(flog, '\n>>> Training for the epoch %d,  %d/%d ...' % ( epoch+1+self.epochContinueNumber, epoch+1, self.TRAINING_EPOCHES))

                train_file_idx = np.arange(0, len(train_file_list))
                np.random.shuffle(train_file_idx)

                train_one_epoch(epoch)
                flog.flush()

                cp_filename = saver.save(sess, os.path.join(self.PRETRAINED_MODEL_PATH, 'epoch_' + str(epoch + 1 + self.epochContinueNumber) + '.ckpt'))
                self.Printout(flog, '\n' + time.strftime("%H:%M:%S", time.localtime()) + ' - Successfully store the checkpoint model into ' + cp_filename)

            flog.close()