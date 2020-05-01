## Description of files
### module.py
Definition of the network architecture in components, i.e. encoder, decoder, Gaussian noise layer, and discriminator.

### train.py
Defines the various losses, the dataset, and the training loop. Performs training.

### test.py
Short script to run model on some test images.

### data.py
Makes the dataset by zipping together real image inputs with (tuples of) synthetic image inputs. Also contains method to crop b1, b2, and b3 from synthetic image tuple B. Handles pre-processing (converting the image tensor to dtype=tf.float32 and normalizing the values to be between 0 and 1).

### process_Xb.py
Crops tuples of synthetic image inputs into individual b1, b2, b3 images and saves them in separate directories. (Ultimately unused in the final product; we considered splitting the synthetic image inputs in this way when we were exploring how the data should be parsed and flow through the network.)

### extract_data.py
Loads data from .tar files and saves extracted images to appropriate directories.


### split_data.py
Splits real input images and (tuples of) synthetic input images randomly into 80% train and 20% test and saves the files in the appropriate directories.

## How to install & run
### Requirements:
- tensorflow==2.1.0
- tensorflow_addons (not supported for Windows)
- numpy
- tqdm
- GPUs

Clone the Github repo, which includes the data in the appropriate directories.

### Example command to train
```
CUDA_VISIBLE_DEVICES=0 python train.py --epochs  300
```

### Example command to test
```
CUDA_VISIBLE_DEVICES=0 python test.py --experiment_dir ./output/digit-data
```
