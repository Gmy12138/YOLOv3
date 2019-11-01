# YOLOv3
#### pytorch 1.3 and python 3.6 is suppoted
A PyTorch implementation of YOLOv3, with support for training, inference and evaluation.

## Introduction
#####The method of yolov3 was used to perform defect detection on NEU surface defect database,and We adopted data enhancement methods such as random clipping, flipping and color enhancement. Finally achieved a satisfactory result.


## Installation
##### Clone and install requirements
    $ git clone https://github.com/Gmy12138/YOLOv3
    $ cd YOLOv3/
    $ sudo pip install -r requirements.txt

##### Download pretrained weights
    $ cd weights/
    $ bash download_weights.sh

##### Download NEU-DET dataset
    Download address    http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
    $ cd data/
    $ Put the data in the NEU-DET dataset folder
    
## Test
Evaluates the model on NEU-DET test.
```
   NDE: Without Data Enhancement    
   DE: Data Enhancement
```

| Model                   | mAP (min. 50 IoU) |
| ----------------------- |:-----------------:|
| YOLOv3 416 (WDE)        | 53.5              |
| YOLOv3 416 (DE)         | 67.1              |
| YOLOv3 320 (DE)         | 65.1              |

## Inference
Uses pretrained weights to make predictions on images. The Darknet-53 measurement marked shows the inference time of this implementation on my 2080ti card.

| Backbone                | GPU      | FPS      |
| ----------------------- |:--------:|:--------:|
| Darknet-53              | 2080ti   |          |


## Train
```
$ train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--gradient_accumulations GRADIENT_ACCUMULATIONS]
                [--model_def MODEL_DEF] [--data_config DATA_CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--compute_map COMPUTE_MAP]
                [--multiscale_training MULTISCALE_TRAINING]
```

#### Training log
```
---- [Epoch 7/100, Batch 8/150] ----
+------------+--------------+--------------+--------------+
| Metrics    | YOLO Layer 0 | YOLO Layer 1 | YOLO Layer 2 |
+------------+--------------+--------------+--------------+
| grid_size  | 16           | 32           | 64           |
| loss       | 1.554926     | 1.446884     | 1.427585     |
| x          | 0.028157     | 0.044483     | 0.051159     |
| y          | 0.040524     | 0.035687     | 0.046307     |
| w          | 0.078980     | 0.066310     | 0.027984     |
| h          | 0.133414     | 0.094540     | 0.037121     |
| conf       | 1.234448     | 1.165665     | 1.223495     |
| cls        | 0.039402     | 0.040198     | 0.041520     |
| cls_acc    | 44.44%       | 43.59%       | 32.50%       |
| recall50   | 0.361111     | 0.384615     | 0.300000     |
| recall75   | 0.222222     | 0.282051     | 0.300000     |
| precision  | 0.520000     | 0.300000     | 0.070175     |
| conf_obj   | 0.599058     | 0.622685     | 0.651472     |
| conf_noobj | 0.003778     | 0.004039     | 0.004044     |
+------------+--------------+--------------+--------------+
Total Loss 4.429395
---- ETA 0:35:48.821929
```

#### Tensorboard
Track training progress in Tensorboard:
* Initialize training
* Run the command below
* Go to http://localhost:6006/

```
$ tensorboard --logdir='logs' --port=6006
```




