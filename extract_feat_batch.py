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
def extract_features(image_paths, bboxes, model):
    model.eval()  # Set the model to evaluation mode
    num_patches = len(image_paths)
    features_list = []

    with torch.no_grad():
        for i in range(num_patches):
            image = Image.open(image_paths[i]).convert('RGB')
            x, y, w, h = bboxes[i]
            x, y, w, h = int(x), int(y), int(w), int(h)
            patch = image.crop((x, y, x + w, y + h))
            patch = transform(patch)
            patch = patch.unsqueeze(0)  # Add a batch dimension
            patch = patch.cuda() if torch.cuda.is_available() else patch.cpu()
            features = model.extract_features(patch)
            features = F.normalize(features, dim=1)  # L2 normalize
            features_list.append(features)

    return torch.cat(features_list, dim=0)

def main():
    # Load your pre-trained model
    model_path = '/home/fatih/phd/Person_reID_baseline_pytorch/model/ft_ResNet50_reid128_noaug_bs32/net_last.pth'
    model = ft_net(linear_num=128)
    model.load_state_dict(torch.load(model_path))  # Load your pre-trained weights
    model = model.cuda() if torch.cuda.is_available() else model.cpu()
    model.eval()

    # Define batch size for processing bounding boxes
    batch_size = 8  # You can adjust this value based on your system's memory capacity

    # Load your dataset and annotation file
    data_dir = '/home/fatih/phd/mot_dataset/SOMPT22/images/train/'
    annotation_file = '/home/fatih/phd/mot_dataset/SOMPT22/annotations/train.json'

    # Read the annotation file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Initialize an empty list to store the updated annotations
    updated_annotations = []

    # Process annotations in batches
    for i in range(0, len(annotations['annotations']), batch_size):
        batch_annotations = annotations['annotations'][i:i + batch_size]  # Get a batch of annotations

        # Lists to store file names and bounding boxes for the batch
        batch_file_names = []
        batch_bboxes = []

        for annotation in batch_annotations:
            image_id = annotation['image_id']
            file_name = os.path.join(data_dir, annotations['images'][image_id - 1]['file_name'])
            bbox = annotation['bbox']

            batch_file_names.append(file_name)
            batch_bboxes.append(bbox)

        # Extract features for the batch
        features_batch = extract_features(batch_file_names, batch_bboxes, model)

        # Update the batch of annotations with the feature vectors
        for j, annotation in enumerate(batch_annotations):
            annotation['embedding'] = features_batch[j].tolist()

        updated_annotations.extend(batch_annotations)

    # Update the original annotations with the updated batch
    annotations['annotations'] = updated_annotations

    # Save the updated annotations back to the JSON file
    updated_annotation_file = '/home/fatih/phd/mot_dataset/SOMPT22/annotations/updated_training_batch.json'
    with open(updated_annotation_file, 'w') as f:
        json.dump(annotations, f)

    # Now, 'updated_training_batch.json' includes the 'embedding' key with feature vectors for each bounding box

if __name__ == "__main__":
    main()
