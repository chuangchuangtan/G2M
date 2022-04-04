"""
"""
import numpy as np
import os
import cv2
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url
from PIL import Image
from torchvision import transforms
from vggnet import VggLoc

model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}
_layers = ['conv5_3', 'conv5_2', 'conv5_1', 'conv5_0', 'pool4', 'conv4_3', 'conv4_2', 'conv4_1', 'pool3', 'conv3_3', 'conv3_2', 'conv3_1', 'pool2', 'conv2_2', 'conv2_1', 'pool1', 'conv1_2', 'conv1_1']
image_path = './sample'


def normalize_scoremap(cam):
    cam -= cam.min()
    cam /= cam.max()+10**-8
    return cam

def heat_it(map,images,image_bet=0.3):
    image_size = images.shape[:2][::-1]
    map_heat = cv2.applyColorMap(np.uint8(normalize_scoremap(cv2.resize(map, image_size, interpolation=cv2.INTER_CUBIC))*255.), cv2.COLORMAP_JET)
    cam_heat_W = cv2.addWeighted(images, image_bet, np.uint8(map_heat), 1 - image_bet, 0)
    return cam_heat_W
def get_targetlayer():
    print(' '.join(sys.argv))
    if len(sys.argv)>=2: target_layer = str(sys.argv[1])
    else: target_layer = "conv5_1"
    assert target_layer in _layers
    return target_layer

def get_model():
    model = VggLoc()
    model.load_state_dict(load_url(model_urls['vgg16'], progress=True), strict=True)
    if torch.cuda.is_available(): model.to('cuda')
    return model
    
    
if __name__ == '__main__':
    
    target_layer = get_targetlayer()
    model = get_model()
    assert os.path.isdir(image_path)
    results_path = os.path.join('./results',target_layer)
    os.makedirs(results_path, exist_ok=True)
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        
    for filename in os.listdir(  image_path):
        input_image = Image.open(os.path.join(image_path,filename))
        image_np = np.array(input_image)
        input_tensor = preprocess(input_image).unsqueeze(0)
        if torch.cuda.is_available(): input_tensor = input_tensor.to('cuda')
        
        
        layercam = model.get_Layercam(input_tensor, target_layer)
        cv2.imwrite(os.path.join(results_path, filename.split('.')[0]+'_layercam.'+filename.split('.')[1]), heat_it(layercam[0],image_np) )
        
        SGLG1 = model.get_SGLG1(input_tensor, target_layer)
        cv2.imwrite(os.path.join(results_path, filename.split('.')[0]+'_SGLG1.'+filename.split('.')[1]), heat_it(SGLG1[0],image_np) )
        
        SGLG2 = model.get_SGLG2(input_tensor, target_layer)
        cv2.imwrite(os.path.join(results_path, filename.split('.')[0]+'_SGLG2.'+filename.split('.')[1]), heat_it(SGLG2[0],image_np) )
        
        DGL = model.get_DGL(input_tensor, target_layer)
        cv2.imwrite(os.path.join(results_path, filename.split('.')[0]+'_DGL.'+filename.split('.')[1]), heat_it(DGL[0],image_np) )


