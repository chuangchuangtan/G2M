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
|pool3     | Layercam   |42.23     |52.72     |56.95                |
|pool3     | sgl-g1     |38.00     |47.64     |51.83                |
|conv3_3   | Layercam   |43.00     |53.51     |57.44                |
|conv3_3   | sgl-g1     |45.06     |56.16     |60.63                |
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

