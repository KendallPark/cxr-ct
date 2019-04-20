import h5py
import time
from keras.models import model_from_json
import numpy as np
import imageio as imio


def test_import_data_1_images(filename1):
    # this function is for testing only

    dict_data = {}
    dict_data['f1'] = imio.imread(filename1)
    dict_data['fT1'] = np.reshape(dict_data['f1'], (256, 256, 1))
    dict_data['fTT1'] = np.expand_dims(dict_data['fT1'], axis=0)

    return dict_data


if __name__ == '__main__':
    #     folder_weight = 'C:\\Users\\Lanston\\Desktop\\training_setting\\weight\\'

    #     filename_weights_import= 'skulls_tf_001b.npy'
    # filename_weights_import= 'skulls_tf_001_2019.04.13_Sat_01.24.52.npy'
    google_bucket_switch = 0

    # record starting time
    time_start = time.time()

    # filename_2D_bucket = 'gs://cxr-to-chest-ct2/training_data/data_2d_image/Round1_2019_04_10/First_Round_Output_Xray_0001_001.png'
    filename_2D_input = 'C:\\Users\\Lanston\\Desktop\\2Dimage1012\\output\\First_Round_Output_Xray_0001_001.png'
    # filename_2D_input = '/tmp/First_Round_Output_Xray_0001_001.png'

    filename_3D_output = 'C:\\Users\\Lanston\\Desktop\\predict0001.h5'
    # filename_3D_output = '/tmp/predict0001.h5'

    weights_import = 'C:\\Users\\Lanston\\Desktop\\training_setting\\weight\\model_weights.h5'
    # weights_import = '/tmp/model_weight/model_weights.h5'

    structure_import = 'C:\\Users\\Lanston\\Desktop\\training_setting\\structure\\model.json'
    # structure_import = '/tmp/model.json'

    2  # !gsutil cp {filename_2D_bucket} {filename_2D_input}

    dict_2_img = test_import_data_1_images(filename_2D_input)

    json_file = open(structure_import, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    with open(structure_import, 'r') as f:
        model = model_from_json(f.read())

    model.load_weights(weights_import)

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy', 'mean_squared_error'])

    out = model.predict(dict_2_img['fTT1'])
    out_cube = out.reshape((128, 128, 128))

    print('Max. value :' + str(np.max(out)))
    print('Min. value :' + str(np.min(out)))
    print('Min. Abs value :' + str(np.min(abs(out))))

    h5f = h5py.File(filename_3D_output, 'w')
    h5f.create_dataset('patient_data', data=out_cube)
    h5f.close()

    # record ending time
    time_end = time.time()

    print("Total running time: " + str(time_end - time_start) + "s")  # in seconds