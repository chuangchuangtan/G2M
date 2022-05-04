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

| Layers   | Method     | Top1 loc | Top5 loc | GT-Known loc        |
| -------- | --------   | ----     | ----     | ----                |
|conv5_3   | Layercam   |44.19     |55.02     |59.29                |
|conv5_3   | sgl-g1     |32.16     |39.90     |42.80        |
|conv5_2   | Layercam   |45.83     |56.88     |61.23                |
|conv5_2   | sgl-g1     |40.16     |50.11     |54.01         |
|conv5_1   | Layercam   |43.69     |54.21     |58.16                |
|conv5_1   | sgl-g1     |47.70     |59.32     |63.87(+5.71)[best]   |
|pool4     | Layercam   |44.10     |54.83     |58.99                |
|pool4     | sgl-g1     |46.81     |58.23     |62.71         |
|conv4_3   | Layercam   |37.95     |47.67     |51.91                |
|conv4_3   | sgl-g1     |53.91     |43.45     |57.82        |
|conv4_2   | Layercam   |42.85     |53.54     |57.89                |
|conv4_2   | sgl-g1     |42.64     |53.39     |58.12         |
|conv4_1   | Layercam   |42.99     |53.57     |57.73                |
|conv4_1   | sgl-g1     |39.08     |49.09     |53.48         |
|pool3     | Layercam   |42.23     |52.72     |56.95                |
|pool3     | sgl-g1     |37.80     |47.35     |51.52         |


# TODO

Release codes about G2M

