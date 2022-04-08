# G2M
Gradual Guidance Learning with Middle Feature for Weakly Supervised Object Localization

=======

# Prerequisites 
- pytorch                   1.8.0
- opencv-contrib-python     4.5.2.54 
- opencv-python             4.5.3.56 
- Pillow                    8.2.0

# Test
Use the test script to generate attention maps of LayerCAM, SGL, DGL on official pytorch VGG model

```
python test_loc.py conv5_1
```
# Results on caffe VGG model using [layercam_loc](https://github.com/PengtaoJiang/layercam_loc).
| Method   | metric              | S5   | S4   | S3   | S2   | S1   |
| -------- | --------            | ---- | ---- | ---- | ---- | ---- |
| layercam | Top1 loc            |46.62 |44.05 |41.83 |43.18 |43.71 |
| layercam | Top5 loc            |57.83 |55.02 |52.28 |53.60 |54.34 |
| layercam | Top1 loc without cla|62.02 |59.48 |55.50 |      |      |
| sgl-g1   | Top1 loc            |      |      |      |      |      |
| sgl-g1   | Top5 loc            |      |      |      |      |      |
| sgl-g1   | Top1 loc without cla|      |      |      |      |      |
| sgl-g3   | Top1 loc            |30.39 |46.54 |44.34 |      |      |
| sgl-g3   | Top5 loc            |37.73 |57.88 |55.02 |      |      |
| sgl-g3   | Top1 loc without cla|40.45 |62.32 |59.21 |      |      |



Rssults on different layers

| Layers   | Method     | Top1 loc | Top5 loc | Top1 loc without cla|
| -------- | --------   | ----     | ----     | ----                |
|conv5_3   | Layercam   |44.19     |55.02     |59.29                |
|conv5_3   | sgl-g1     |32.16     |39.90     |42.80                |
|conv5_2   | Layercam   |45.83     |56.88     |61.23                |
|conv5_2   | sgl-g1     |38.75     |48.35     |52.12                |
|conv5_1   | Layercam   |43.69     |54.21     |58.16                |
|conv5_1   | sgl-g1     |47.36     |58.85     |63.37                |
|pool4     | Layercam   |44.10     |54.83     |58.99                |
|pool4     | sgl-g1     |46.55     |57.86     |62.29                |
|conv4_3   | Layercam   |37.95     |47.67     |51.91                |
|conv4_3   | sgl-g1     |28.64     |20.89     |26.29                |
|conv4_2   | Layercam   |42.85     |53.54     |57.89                |
|conv4_2   | sgl-g1     |42.71     |53.49     |58.22                |
|conv4_1   | Layercam   |42.99     |53.57     |57.73                |
|conv4_1   | sgl-g1     |39.35     |49.41     |53.83                |
|pool3     | Layercam   |          |          |                     |
|pool3     | sgl-g1     |          |          |                     |
|conv3_3   | Layercam   |          |          |                     |
|conv3_3   | sgl-g1     |          |          |                     |
|conv3_2   | Layercam   |          |          |                     |
|conv3_2   | sgl-g1     |          |          |                     |
|conv3_1   | Layercam   |          |          |                     |
|conv3_1   | sgl-g1     |          |          |                     |
|pool2     | Layercam   |          |          |                     |
|pool2     | sgl-g1     |          |          |                     |
|conv2_2   | Layercam   |          |          |                     |
|conv2_2   | sgl-g1     |          |          |                     |
|conv2_1   | Layercam   |          |          |                     |
|conv2_1   | sgl-g1     |          |          |                     |
|pool1     | Layercam   |          |          |                     |
|pool1     | sgl-g1     |          |          |                     |
|conv1_2   | Layercam   |          |          |                     |
|conv1_2   | sgl-g1     |          |          |                     |
|conv1_1   | Layercam   |          |          |                     |
|conv1_1   | sgl-g1     |          |          |                     |

# TODO

Release codes about G2M

