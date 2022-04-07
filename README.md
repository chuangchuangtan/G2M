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
| Method   | metric   | S5   | S4   | S3   | S2   | S1   |
| -------- | -------- | ---- | ---- | ---- | ---- | ---- |
| layercam | Top1 loc |46.62 |44.05 |41.83 |43.18 |43.71 |
| layercam | Top1 loc |57.83 |55.02 |52.28 |53.60 |54.34 |
| sgl-g11  | Top1 loc |      |      |      |      |      |
| sgl-g1   | Top1 loc |      |      |      |      |      |




# TODO

Release codes about G2M

