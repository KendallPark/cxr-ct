import time
import numpy as np
import imageio as imio
import h5py
import os

# Google bucket
folder_bucket_2D_prefix = 'gs://cxr-to-chest-ct2/training_data/data_2d_image/Round1_2019_04_10/'  # actual folder
folder_bucket_3D_prefix = 'gs://cxr-to-chest-ct2/training_data/data_3d_scan/'  # actual folder
# folder_bucket_2D_prefix = 'gs://cxr-to-chest-ct2/training_data/zz_temp_data_2d_image/' # testing folder
# folder_bucket_3D_prefix = 'gs://cxr-to-chest-ct2/training_data/zz_temp_data_3d_scan/' # testing folder
folder_bucket_weight_prefix = 'gs://cxr-to-chest-ct2/training_data/model_weight/'
folder_bucket_weight_backup_prefix = 'gs://cxr-to-chest-ct2/training_data/model_weight/backup/'
folder_bucket_log_prefix = 'gs://cxr-to-chest-ct2/training_data/training_log/'
folder_bucket_log_backup_prefix = 'gs://cxr-to-chest-ct2/training_data/training_log/00Backup/'
folder_bucket_structure_prefix = 'gs://cxr-to-chest-ct2/training_data/model_structure/'

# 2D input
# folder_data_2D = 'C:\\Users\\Lanston\\Desktop\\2Dimage1012\\testing\\2D\\'
# folder_data_2D = '/tmp/zz_test_2D/'
folder_data_2D = '2D/'
# filename_prefix='Output_Xray_'
filename_prefix = 'First_Round_Output_Xray_'
filename_suffix = '.png'

# 3D input
# folder_data_3D = 'C:\\Users\\Lanston\\Desktop\\2Dimage1012\\testing\\3D\\'
# folder_data_3D = '/tmp/zz_test_3D/'
folder_data_3D = '3D/'
name_database_3D = 'patient_data'

# weight
# folder_weight = 'C:\\Users\\Lanston\\Desktop\\training_setting\\weight\\'
folder_weight = 'model_weight/'

filename_weights_import = 'model_weights.h5'
filename_weights_export = 'model_weights.h5'
filename_weights_export_backup_prefix = 'model_weights01_'

# folder_structure = 'C:\\Users\\Lanston\\Desktop\\training_setting\\structure\\'
folder_structure = 'model_structure/'
filename_structure_import = 'model01.json'
filename_structure_export = 'model01.json'

# log
# folder_log = 'C:\\Users\\Lanston\\Desktop\\training_setting\\log\\'
folder_log = 'log/'
timezone_adj = 5 * 60 * 60  # summer time for Chicago
filename_log_export_backup_prefix = 'log_'

# batch size for training
batch = 5

# number of batch
# num_batch=4
num_batch = 1500
num_run = batch * num_batch

google_bucket_switch = 1  # = 0 or 1; if = 1, then will import from/export to bucket.
backup_freq = 50  # special weights backup to backup folder with date-time-filename every k batches; -1 if don't need

# number of patient; patient238 and patient 585 don't exist
num_patient = 1012

# train/test split
freq_removal = 10  # freq of reserving data for test set

# number of combination (translation/rotation/rescaling etc.) for each image
# num_combine=8*8*8
num_combine = 1  # use this before the 2D images combinations are available


def import_2D_img(folder_bucket_2D_prefix, folder_data_2D, filename_prefix, \
                  filename_suffix, idx_img_all, idx_combine_all, batch, j_batch, bucket_flag):
    # this function will import 2D images from (bucket)/local
    # bucket_flag = 0 or 1; if bucket_flag=1, then import file from bucket to local

    # initialize stack_2d_img
    stack_2d_img = np.array([], dtype=np.int64).reshape(0, 256, 256, 1)

    for i in range(batch):
        idx_img = idx_img_all[(j_batch - 1) * batch + i]
        idx_combine = idx_combine_all[(j_batch - 1) * batch + i]

        filename_2D = filename_prefix + ("%04d" % idx_img) + '_' \
                      + ("%03d" % idx_combine) + filename_suffix

        ### import from Google Bucket ###
        # if bucket_flag == 1:
        #     path_bucket = folder_bucket_2D_prefix + filename_2D
        #     path_local = folder_data_2D + filename_2D
        #     if os.path.isfile(path_local):
        #         print(filename_2D + ' is already in local.')
        #     else:
        #         # !gsutil cp {path_bucket} {path_local}
        #
        # elif bucket_flag == 0:
        #       # do nothing
        # else:
        #     print('error in import_2D_img()')

            ### import from local ###
        filepath_2D = folder_data_2D + filename_2D
        f1 = imio.imread(filepath_2D)
        fTT1 = np.reshape(f1, (1, 256, 256, 1))

        stack_2d_img = np.vstack([stack_2d_img, fTT1])

    return stack_2d_img


def import_3D_scan(folder_bucket_3D_prefix, folder_data_3D, \
                   name_database_3D, idx_img_all, batch, j_batch, bucket_flag):
    # this function will import 3D scan from (bucket)/local
    # for reading .h5 file only; may need to be modified if .npy or other format is used
    # bucket_flag = 0 or 1; if bucket_flag=1, then import file from bucket to local

    # initialize stack_2d_img
    stack_3d_scan = np.array([], dtype=np.int64).reshape(0, 128 ** 3)

    for i in range(batch):
        idx_img = idx_img_all[(j_batch - 1) * batch + i]
        filename_3D = ("%04d" % idx_img) + '.h5'

        ### import from Google Bucket ###
        # if bucket_flag == 1:
        #     path_bucket = folder_bucket_3D_prefix + filename_3D
        #     path_local = folder_data_3D + filename_3D
        #     if os.path.isfile(path_local):
        #         print(filename_3D + ' is already in local.')
        #
        # elif bucket_flag == 0:
        #     2  # do nothing
        # else:
        #     print('error in import_3D_img()')

            ### import from local ###
        filepath_3D = folder_data_3D + filename_3D
        file_3D = h5py.File(filepath_3D, 'r')
        data_3D = np.array(file_3D[name_database_3D])

        data_3D_T = np.reshape(data_3D, (1, 128 ** 3))

        stack_3d_scan = np.vstack([stack_3d_scan, data_3D_T])

    return stack_3d_scan


def import_weights(model, path_import, initial_flag, bucket_flag):
    # input: initial_flag=1 if it is the importation from the original .npy file
    # input: initial_flag=0 for the subsequent training
    # bucket_flag = 0 or 1; if bucket_flag=1, then import file from bucket to local

    model.load_weights(path_import)

    return model
    # add the pre-existing weights to the model


#     np_weights = np.load(path_import, encoding='latin1')
#     layers = np_weights[()].keys()
#     # key_weight_template={'weights':999}.keys()
#     # key_weight_template=np_weights[()]['conv1_b'].keys()
#     # bn_weight=np_weights[()]['bn_conv1_b'].keys()

#     for layer in model.layers:
#         if layer.name in layers:
#             # only upload for layers with weight

#             if initial_flag==1:
#                 if len(np_weights[()][layer.name].keys()) == 1:
#                 #print(layer.name)
#                     weight = np_weights[()][layer.name]['weights']


#                     # fix discrepancies on 6 layers
#                     if layer.name in ['hg1_low6_low6_low6_up5', 'hg1_low6_low6_up5', 'hg1_low6_up5', 'hg1_up5']:
#                       weight = np.repeat(weight, 256, axis=2)
#                       layer.set_weights([weight, np.zeros(256)])
#                     elif layer.name in ['res1_branch2a', 'res6_branch2a']:
#                       weight = np.repeat(weight, 2, axis=2)
#                       layer.set_weights([weight])
#                     else:
#                       layer.set_weights([weight])


#                 # add weights to the batch normalization layers
#                 elif len(np_weights[()][layer.name].keys()) == 4: # no need to compare arrays like we did above,
#                                                                   # this is more efficient
#                     weight_dict = np_weights[()][layer.name]

#                     #case: tensor rank 4 (original .npy file; not necessary; less efficient)
#                     scale = weight_dict['scale'][0][0][0]
#                     offset = weight_dict['offset'][0][0][0]
#                     mean = weight_dict['mean'][0][0][0]
#                     variance = weight_dict['variance'][0][0][0]
#                     layer.set_weights([scale, offset, mean, variance])

#                 # add weights to the deconv layers
#                 elif len(np_weights[()][layer.name].keys()) == 2:
#                     2 # do nothing; already done above

#             elif initial_flag==0:
#                 if len(np_weights[()][layer.name].keys()) == 1:
#                     weight = np_weights[()][layer.name]['weights']
#                     layer.set_weights([weight])
#                 elif len(np_weights[()][layer.name].keys()) == 4:
#                     weight_dict = np_weights[()][layer.name]
#                     #case: vector rank 1 (training .npy file; more efficient)
#                     scale = weight_dict['scale']
#                     offset = weight_dict['offset']
#                     mean = weight_dict['mean']
#                     variance = weight_dict['variance']

#                 # add weights to the deconv layers
#                 elif len(np_weights[()][layer.name].keys()) == 2:
#                     weight_dict = np_weights[()][layer.name]
#                     weight = weight_dict['weights']
#                     bias = weight_dict['bias']
#                     layer.set_weights([weight, bias])

#             else:
#                 print('error in import_weights()')


def export_weights(model, path_export, bucket_flag, j_batch, time_end_j, backup_freq):
    #     num_layers=len(model.layers)
    #     # key_weight_template={'weight':999}.keys()
    #     # input: e.g. bucket_flag = 0 or 1; if bucket_flag=1, then import file from bucket to local
    #     # input: e.g. if backup_freq = 7, then will export weights to Google bucket for every 7 batches
    #     # input: backup_freq = -1 if we don't want to backup

    #     # initialize empty dictioinary
    #     dict={}
    #     for i in range(num_layers):
    #         layer_name=model.layers[i].get_config()['name']
    #         num_weight=len(model.layers[i].get_weights())
    #         # print(layer_name+'_'+str(num_weight))
    #         if num_weight==1: # weight for conv layer without bias
    #             dict_i={'weights':model.layers[i].get_weights()[0]}
    #             dict.update({layer_name: dict_i})
    #         elif num_weight==2: # weight for deconv layer with bias
    #             dict_i={'weights':model.layers[i].get_weights()[0], \
    #                     'bias':model.layers[i].get_weights()[1]}
    #             dict.update({layer_name: dict_i})
    #         elif num_weight==4: # weight for BN layer
    #             dict_i={'scale':model.layers[i].get_weights()[0], \
    #                     'offset':model.layers[i].get_weights()[1], \
    #                     'mean':model.layers[i].get_weights()[2], \
    #                     'variance':model.layers[i].get_weights()[3]}
    #             dict.update({layer_name: dict_i})

    #         elif num_weight==0:
    #             2 #do nothing
    #             # dict_i='Nil'

    #     array=np.array(dict)

    #     np.save(path_export, array)

    model.save_weights(path_export)

    # ### export to Google Bucket ###
    # if bucket_flag == 1:
    #     path_local = path_export
    #     path_bucket = folder_bucket_weight_prefix
    #     # !gsutil cp {path_local} {path_bucket}
    #
    #     # backup every k batches
    #     if (j_batch % backup_freq) == (backup_freq - 1):
    #         time_end_file_str = time.strftime("%Y.%m.%d_%a_%H.%M.%S", time.gmtime(time_end_j))
    #         path_local = path_export
    #         path_bucket = folder_bucket_weight_backup_prefix + filename_weights_export_backup_prefix \
    #                       + time_end_file_str + '.h5'
    #         # !gsutil cp {path_local} {path_bucket}
    #
    #
    # elif bucket_flag == 0:
    #     2  # do nothing
    # else:
    #     print('error in export_weights()')

    return


def load_structure(model, path_import, bucket_flag):
    # if bucket_flag == 1:
    #     path_bucket = folder_bucket_structure_prefix + filename_structure_import
    #     path_local = folder_structure
    #     # !gsutil cp {path_bucket} {path_local}
    #
    # elif bucket_flag == 0:
    #     2  # do nothing
    # else:
    #     print('error in load_structure()')

    json_file = open(path_import, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    with open(path_import, 'r') as f:
        model = model_from_json(f.read())

    return model


def create_log(batch, num_batch, j_batch, time_start, time_end, folder_log, \
               msg_history, bucket_flag, time_end_j, backup_freq):
    # bucket_flag = 0 or 1; if bucket_flag=1, then import file from bucket to local

    msg_training = 'Training done:- ' + '\n' \
                   + 'batch size = ' + str(batch) + ';' + '\n' \
                   + 'j-th batch = ' + str(j_batch + 1) + '/ num_batch = ' + str(num_batch) + '; ' + '\n' + '\n'

    time_start_str = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime(time_start))
    time_end_str = time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime(time_end))
    time_end_file_str = time.strftime("%Y.%m.%d_%a_%H.%M.%S", time.gmtime(time_end))

    msg_time = 'Time started: ' + time_start_str + ';' + '\n' + 'Time end: ' + time_end_str + ';' + '\n' + 'Time used: ' + (
                "%.2f" % (time_end - time_start)) + 's; ' + '\n' + '\n'
    msg_hist_head = 'Training history is as below:- ' + '\n'
    msg_log = msg_training + msg_time + msg_hist_head + msg_history

    filename_log = 'log.txt'
    filepath_log = folder_log + filename_log
    file = open(filepath_log, "w+")
    file.write(msg_log)
    file.close()

    ### export to Google Bucket ###
    # if bucket_flag == 1:
    #     path_local = folder_log + filename_log
    #     path_bucket = folder_bucket_log_prefix
    #     # !gsutil
    #     # cp
    #     # {path_local}
    #     # {path_bucket}
    #     # backup every k batches
    #     if (j_batch % backup_freq) == (backup_freq - 1):
    #         time_end_file_str = time.strftime("%Y.%m.%d_%a_%H.%M.%S", time.gmtime(time_end_j))
    #         path_local = folder_log + filename_log
    #         path_bucket = folder_bucket_log_backup_prefix + filename_log_export_backup_prefix \
    #                       + time_end_file_str + '.txt'
    #         # !gsutil
    #         # cp
    #         # {path_local}
    #         # {path_bucket}
    #
    # elif bucket_flag == 0:
    #     2  # do nothing
    # else:
    #     print('error in create_log()')

    return


if __name__ == '__main__':

    time_start = time.time()

    # remove non-existent patient238 and patient585;
    # reserve patient for testing
    patient_non_exist = [238, 585]
    patient_option_test = list(range(freq_removal, num_patient + 1, freq_removal))
    patient_option_remove = patient_option_test + patient_non_exist + [0]
    patient_option_train = np.delete(list(range(num_patient + 1)), patient_option_remove)
    idx_img_all = np.random.choice(patient_option_train, num_run)

    # combine_option=np.delete(np.arange(num_combine)+1,np.arange(freq_removal,num_combine,freq_removal))
    combine_option = np.arange(num_combine) + 1
    idx_combine_all = np.random.choice(combine_option, num_run)

    #######################
    # hide these 4 lines in actual run
    # idx_img_all=np.array([950, 427, 124, 932, 7])
    # idx_combine_all=np.array([1, 1, 1, 1, 1])

    # num_batch=1
    # batch=5
    #######################

    # import weights
    model = import_weights(model, folder_weight + filename_weights_import, 0, google_bucket_switch)

    # load structure
    model = load_structure(model, folder_structure + filename_structure_import, google_bucket_switch)

    # compile model
    learning_rate = 0.001
    b1 = 0.9
    b2 = 0.999
    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=b1, beta_2=b2)

    model.compile(optimizer=adam,
                  loss='mean_squared_error',
                  metrics=['accuracy', 'mean_squared_error'])

    time_after_import_weights = time.time()
    print("Import Weights time: " + str(time_after_import_weights - time_start) + "s")  # in seconds

    # initialize msg_history
    msg_history = ''
    for j_batch in range(num_batch):
        ### import 2D images ###
        time_before_import_2D_j = time.time()
        stack_2d_img = import_2D_img(folder_bucket_2D_prefix, folder_data_2D, \
                                     filename_prefix, filename_suffix, idx_img_all, \
                                     idx_combine_all, batch, j_batch, google_bucket_switch)
        time_after_import_2D_j = time.time()

        ### import 3D scan ###
        stack_3d_scan = import_3D_scan(folder_bucket_3D_prefix, folder_data_3D, \
                                       name_database_3D, idx_img_all, batch, j_batch, google_bucket_switch)

        time_after_import_3D_j = time.time()
        # hist=model.fit(stack_2d_img, stack_3d_scan, epochs=2)
        # hist=model.fit(train_images, train_labels, epochs=2) # for testing
        hist = model.fit(stack_2d_img, stack_3d_scan, epochs=1)
        msg_history = msg_history + repr(hist.history) + '\n'  # training progress record
        time_after_training_j = time.time()

        # time zone adjustment;
        time_start_j_adj = time_before_import_2D_j - timezone_adj
        time_end_j_adj = time_after_training_j - timezone_adj

        export_weights(model, folder_weight + filename_weights_export, \
                       google_bucket_switch, j_batch, time_end_j_adj, backup_freq)

        time_after_export_weights_j = time.time()
        time_after_export_weights_j_adj = time_after_export_weights_j - timezone_adj

        create_log(batch, num_batch, j_batch, time_start_j_adj, time_after_export_weights_j_adj, \
                   folder_log, msg_history, google_bucket_switch, \
                   time_after_export_weights_j_adj, backup_freq)

        print("Import 2D time: " + str(time_after_import_2D_j - time_before_import_2D_j) + "s")  # in seconds
        print("Import 3D time: " + str(time_after_import_3D_j - time_after_import_2D_j) + "s")  # in seconds
        print("Training time: " + str(time_after_training_j - time_after_import_3D_j) + "s")  # in seconds
        print("Export weights time: " + str(time_after_export_weights_j - time_after_training_j) + "s")  # in seconds
        print(str(j_batch + 1) + "-th batch of " + str(num_batch) + " batches is completed. \n \n")  # in seconds

    time_end = time.time()

    print("Total run time: " + str(time_end - time_start) + "s")  # in seconds