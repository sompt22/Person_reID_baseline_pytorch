import os
import json
import torch
from torchvision import datasets, models, transforms
from PIL import Image
import numpy as np
import math
from model import ft_net
import torch.nn.functional as F
from collections import defaultdict

# Define a transform to preprocess images
transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class FeatureExtractionDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Group annotations by image
        self.grouped_annotations = defaultdict(list)
        for annotation in annotations['annotations']:
            image_id = annotation['image_id']
            self.grouped_annotations[image_id].append(annotation)

        self.image_ids = list(self.grouped_annotations.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        file_name = os.path.join(self.data_dir, annotations['images'][image_id - 1]['file_name'])

        image = Image.open(file_name).convert('RGB')
        bboxes = [anno['bbox'] for anno in self.grouped_annotations[image_id]]
        return image, bboxes, image_id

def extract_features_from_batch(images, all_bboxes, model):
    extracted_features = []
    image_indices = []
    
    for idx, (image, bboxes) in enumerate(zip(images, all_bboxes)):
        for bbox in bboxes:
            x, y, w, h = bbox
            x, y, w, h = map(int, [math.floor(i) for i in [x, y, w, h]])
            patch = image.crop((x, y, x + w, y + h))
            patch = transform(patch).unsqueeze(0).cuda()
            extracted_features.append(patch)
            image_indices.append(idx)

    extracted_features = torch.cat(extracted_features, dim=0)
    with torch.no_grad():
        features = model.extract_features(extracted_features)
    features = F.normalize(features, dim=1).cpu().numpy()

    return features, image_indices

def main():
    # Load your pre-trained model
    model_path = '/home/fatih/phd/Person_reID_baseline_pytorch/model/ft_ResNet50_reid128_noaug_bs32/net_last.pth'
    model = ft_net(linear_num=128)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.cuda()

    # Load your dataset and annotation file
    data_dir = '/home/fatih/phd/mot_dataset/SOMPT22/images/train/'
    annotation_file = '/home/fatih/phd/mot_dataset/SOMPT22/annotations/train.json'

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    dataset = FeatureExtractionDataset(annotations, data_dir, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)  # Adjust batch size as needed

    for images, all_bboxes, image_ids in dataloader:
        features, image_indices = extract_features_from_batch(images, all_bboxes, model)

        for feat, idx in zip(features, image_indices):
            img_id = image_ids[idx].item()
            annotation = [anno for anno in annotations['annotations'] if anno['image_id'] == img_id][idx]
            annotation['embedding'] = feat.tolist()

    # Save the updated annotations back to the JSON file
    updated_annotation_file = '/home/fatih/phd/mot_dataset/SOMPT22/images/annotations/updated_training.json'
    with open(updated_annotation_file, 'w') as f:
        json.dump(annotations, f)

if __name__ == "__main__":
    main()

