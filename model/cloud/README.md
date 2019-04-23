## How to run job in Google's AI Platform

### Dependencies

- need to [install `gcloud` utility](https://cloud.google.com/sdk/install)
- need to authenticate with an account that access to `x-ray-reconstruction` project and `cxr-to-chest-ct2` bucket.

### Submit training job (in local terminal)

- in local terminal, make sure you're in the `./cxr-ct/model/cloud` directory

### Example training job:

```
gcloud ml-engine jobs submit training my_job_name \
--module-name train.task --package-path train \
--staging-bucket 'gs://cxr-to-chest-ct2/gcp-training/staging' \
--python-version 3.5 --runtime-version 1.13
--packages packages/Keras-2.2.4.tar.gz,packages/Keras-2.2.4-py3-none-any.whl

```

### Example training job (version 2):
# please run the code in the local terminal
# version 2 is tested by Lanston under Windows 10

```

gcloud ml-engine jobs submit training try_21 --config=config.yaml --module-name train.task --package-path train --staging-bucket gs://cxr-to-chest-ct2 --packages packages/Keras-2.2.4.tar.gz,packages/Keras-2.2.4-py3-none-any.whl --region us-central1

```

### Using different datasets

- change the `data-dir` variable in `config.yaml` to change the dataset used.

Examples:

- `tfrecords/big-dobie-2019-04-21--11h59m35s` has all 24 cube rotations (with eval set)
- `tfrecords/silly-cat-2019-04-20--23h57m53s` has one image for each CT scan (no eval set; to use this you will need to change the code in task.py)

### Make stuff faster

https://cloud.google.com/ml-engine/docs/tensorflow/machine-types

### Prediction

- need to investigate

https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-keras
