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



# TODO

Release codes about G2M

