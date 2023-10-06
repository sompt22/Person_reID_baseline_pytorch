import os
import json
import torch
from torchvision import datasets, models, transforms
from PIL import Image
import numpy as np
import math
from model import ft_net, ft_net_dense, ft_net_hr, ft_net_swin, ft_net_swinv2, ft_net_efficient, ft_net_NAS, ft_net_convnext, PCB, PCB_test
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')


opt = parser.parse_args()
###load config###
# load the training config

config_path = os.path.join('./model','ft_ResNet50_reid128_noaug_bs32','opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader) # for the new pyyaml via 'conda install pyyaml'


opt.nclasses = 751

str_ids = opt.gpu_ids.split(',')

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True
    
data_dir = test_dir   

# Load your pre-trained model
model_path = '/home/fatih/phd/Person_reID_baseline_pytorch/model/ft_ResNet50_reid128_noaug_bs32/net_last.pth'
model = ft_net(linear_num=128)
model.load_state_dict(torch.load(model_path))  # Load your pre-trained weights
model.eval()

# Define a transform to preprocess images
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define a function to extract features from a bounding box patch
def extract_features(image_path, bbox):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    
    # Extract bounding box coordinates [x, y, width, height]
    x, y, w, h = bbox
    
    # Crop the bounding box patch from the image
    patch = image[:, y:y+h, x:x+w].unsqueeze(0)  # Add a batch dimension
    
    with torch.no_grad():
        features = model.extract_features(patch)
    
    features = F.normalize(features, dim=1)
    # Extract the 128-dimensional feature vector
    feature_vector = features.squeeze().numpy()
    
    return feature_vector

# Load your dataset and annotation file
data_dir = '/home/fatih/phd/mot_dataset/SOMPT22/images/train/'
annotation_file = '/home/fatih/phd/mot_dataset/SOMPT22/annotations/train.json'

# Read the annotation file
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Update annotations with feature vectors for each bounding box
for annotation in annotations['annotations']:
    image_id = annotation['image_id']
    file_name = os.path.join(data_dir, annotations['images'][image_id - 1]['file_name'])  # Find the corresponding image file
    bbox = annotation['bbox']  # Extract bounding box coordinates [x, y, width, height]
    features = extract_features(file_name, bbox)
    
    # Add the feature vector to the annotation under the "embedding" key
    annotation['embedding'] = features.tolist()

# Save the updated annotations back to the JSON file
updated_annotation_file = '/home/fatih/phd/mot_dataset/SOMPT22/annotations/updated_training.json'
with open(updated_annotation_file, 'w') as f:
    json.dump(annotations, f)

# Now, 'updated_training.json' includes the 'embedding' key with feature vectors for each bounding box
