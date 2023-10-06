import os
import json
import torch
from torchvision import datasets, models, transforms
from PIL import Image
import numpy as np
import math
from model import ft_net
import torch.nn.functional as F
import yaml

# Define a transform to preprocess images
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define a function to extract features from a bounding box patch
def extract_features(image_path, bbox, model):
    # Extract bounding box coordinates [x, y, width, height]
    x, y, w, h = bbox
    x = int(math.floor(x))
    y = int(math.floor(y))
    w = int(math.floor(w))
    h = int(math.floor(h))
    #print(x, y, w, h)

    image = Image.open(image_path).convert('RGB')
    image = image.crop((x, y, x + w, y + h))  # Crop the bounding box from the image
    patch = transform(image)

    # Move the patch and model to the GPU
    patch = patch.cuda()
    model = model.cuda()

    # Crop the bounding box patch from the image
    patch = patch.unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        features = model.extract_features(patch)

    features = F.normalize(features, dim=1)
    # Extract the 128-dimensional feature vector
    feature_vector = features.squeeze().cpu().numpy()  # Move the feature vector back to CPU
    #print(feature_vector.shape)
    #print(feature_vector)

    return feature_vector

def main():
    # Load your pre-trained model
    model_path = '/home/fatih/phd/Person_reID_baseline_pytorch/model/ft_ResNet50_reid128_noaug_bs32/net_last.pth'
    model = ft_net(linear_num=128)
    model.load_state_dict(torch.load(model_path))  # Load your pre-trained weights
    model.eval()
    
    print(model)

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
        print(file_name)
        bbox = annotation['bbox']  # Extract bounding box coordinates [x, y, width, height]
        features = extract_features(file_name, bbox, model)

        # Add the feature vector to the annotation under the "embedding" key
        annotation['embedding'] = features.tolist()

    # Save the updated annotations back to the JSON file
    updated_annotation_file = '/home/fatih/phd/mot_dataset/SOMPT22/images/annotations/updated_training.json'
    with open(updated_annotation_file, 'w') as f:
        json.dump(annotations, f)

    # Now, 'updated_training.json' includes the 'embedding' key with feature vectors for each bounding box

if __name__ == "__main__":
    main()