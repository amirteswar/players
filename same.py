##========================================================================================================
import torch
import os
from torchvision import transforms
from PIL import Image
from model import DogSimilarityClassifier
from cnn import ResNetFeatureExtractor
import torch.nn.functional as F
import time

# Record start time
start_time = time.time()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the feature extractor
feature_extractor = ResNetFeatureExtractor().to(device)

# Load the feature extractor state dict
print("Loading feature extractor state_dict with strict=False")
feature_extractor_state_dict = torch.load("models/resnet_feature_extractor.pth")
filtered_feature_extractor_state_dict = {
    k: v for k, v in feature_extractor_state_dict.items() 
    if k in feature_extractor.state_dict()
}
feature_extractor.load_state_dict(filtered_feature_extractor_state_dict, strict=False)
feature_extractor.eval()

# Initialize the classifier with the loaded feature extractor
model = DogSimilarityClassifier(feature_extractor).to(device)

# Load the classifier state dict
print("Loading classifier state_dict with strict=False")
classifier_state_dict = torch.load("models/dog_similarity_resnet50_triplet.pth")
filtered_classifier_state_dict = {
    k: v for k, v in classifier_state_dict.items() 
    if k in model.state_dict()
}
model.load_state_dict(filtered_classifier_state_dict, strict=False)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the dataset directory
dataset_dir = r'E:\gpu\Sphase\samedog6\data\football_database'

def load_and_transform_image(image_path):
    """Load and transform an image."""
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

def find_similar_image(image_path):
    """Find the most similar image in the dataset to the given query image."""
    query_image = load_and_transform_image(image_path)
    best_similarity = -1
    best_image_path = None

    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                dataset_image_path = os.path.join(root, file)
                dataset_image = load_and_transform_image(dataset_image_path)
                
                with torch.no_grad():
                    query_features = model.features(query_image)
                    dataset_features = model.features(dataset_image)
                    similarity = F.cosine_similarity(query_features, dataset_features).item()
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_image_path = dataset_image_path
    
    return best_image_path, best_similarity

def batch_compute_similarity(query_image_path, dataset_dir, batch_size=32):
    """Compute similarity scores in batches for efficiency."""
    query_image = load_and_transform_image(query_image_path)
    image_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(dataset_dir)
        for file in files if file.endswith(('jpg', 'jpeg', 'png'))
    ]

    best_similarity = -1
    best_image_path = None

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = [load_and_transform_image(p) for p in batch_paths]
        batch_images = torch.cat(batch_images, dim=0)

        with torch.no_grad():
            query_features = model.features(query_image)
            batch_features = model.features(batch_images)
            batch_similarity = F.cosine_similarity(query_features, batch_features).cpu().numpy()
        
        max_similarity_index = batch_similarity.argmax()
        if batch_similarity[max_similarity_index] > best_similarity:
            best_similarity = batch_similarity[max_similarity_index]
            best_image_path = batch_paths[max_similarity_index]

    return best_image_path, best_similarity


# Example usage
query_image_path = r'E:\gpu\Sphase\samedog6\data\test\test_1.jpg'
similar_image_path, similarity_score = batch_compute_similarity(query_image_path, dataset_dir)
print(f"Most similar image path: {similar_image_path}")
print(f"Similarity score: {similarity_score}")

# Record end time
end_time = time.time()
total_time = end_time - start_time
print(f"Total training time: {total_time:.2f} seconds")
