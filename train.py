import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random
from run import DogSimilarityClassifier, ResNetFeatureExtractor
import time

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define custom dataset class for flat directory
class CustomDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        for file in os.listdir(root_dir):
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
                self.image_paths.append(os.path.join(root_dir, file))
        
        if not self.image_paths:
            raise ValueError(f"No images found in directory: {root_dir}")

        self.labels = [0] * len(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        # Handle transparency in images
        if image.mode == 'P':
            image = image.convert("RGBA")
            background = Image.new("RGBA", image.size, (255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")
        else:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Define Triplet Loss
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        positive_distance = torch.nn.functional.pairwise_distance(anchor, positive)
        negative_distance = torch.nn.functional.pairwise_distance(anchor, negative)
        loss = torch.nn.functional.relu(positive_distance - negative_distance + self.margin)
        return loss.mean()

# Create triplets from your data
def generate_triplets(dataset, batch_size=32):
    dataset_size = len(dataset)
    triplets = []

    while len(triplets) < dataset_size:
        indices = random.sample(range(dataset_size), 3)
        anchor, positive, negative = dataset[indices[0]][0], dataset[indices[1]][0], dataset[indices[2]][0]
        triplets.append((anchor, positive, negative))

    return DataLoader(triplets, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

if __name__ == '__main__':
    # Define the root directory containing dog images
    root_dir = r"E:\gpu\Sphase\samedog6\data\football_database"

    # Load dataset
    train_dataset = CustomDogDataset(root_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # Instantiate the ResNet model
    feature_extractor = ResNetFeatureExtractor().to(device)
    model = DogSimilarityClassifier(feature_extractor).to(device)

    # Define loss function and optimizer
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Record start time
    start_time = time.time()

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for triplet in generate_triplets(train_dataset, batch_size=32):
            anchor, positive, negative = triplet
            anchor, positive, negative = anchor.to(device, non_blocking=True), positive.to(device, non_blocking=True), negative.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                anchor_out = model(anchor)
                positive_out = model(positive)
                negative_out = model(negative)
                loss = criterion(anchor_out, positive_out, negative_out)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * anchor.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Record end time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/dog_similarity_resnet50_triplet.pth")
    torch.save(feature_extractor.state_dict(), "models/resnet_feature_extractor.pth")


##======================================================================================================================================================
# # train.py(with time)
# import torch
# import os
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
# import random
# from run import DogSimilarityClassifier, ResNetFeatureExtractor
# import time

# # Define device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Define transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Define custom dataset class for flat directory
# class CustomDogDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = []

#         for file in os.listdir(root_dir):
#             if file.lower().endswith((".jpg", ".jpeg", ".png")):
#                 self.image_paths.append(os.path.join(root_dir, file))
        
#         if not self.image_paths:
#             raise ValueError(f"No images found in directory: {root_dir}")

#         self.labels = [0] * len(self.image_paths)

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         label = self.labels[idx]
#         return image, label

# # Define Triplet Loss
# class TripletLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(TripletLoss, self).__init__()
#         self.margin = margin

#     def forward(self, anchor, positive, negative):
#         positive_distance = torch.nn.functional.pairwise_distance(anchor, positive)
#         negative_distance = torch.nn.functional.pairwise_distance(anchor, negative)
#         loss = torch.nn.functional.relu(positive_distance - negative_distance + self.margin)
#         return loss.mean()

# # Create triplets from your data
# def generate_triplets(dataset, batch_size=32):
#     dataset_size = len(dataset)
#     triplets = []

#     while len(triplets) < dataset_size:
#         indices = random.sample(range(dataset_size), 3)
#         anchor, positive, negative = dataset[indices[0]][0], dataset[indices[1]][0], dataset[indices[2]][0]
#         triplets.append((anchor, positive, negative))

#     return DataLoader(triplets, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# if __name__ == '__main__':
#     # Define the root directory containing dog images
#     root_dir = r"E:\gpu\Sphase\samedog6\data\football_database"

#     # Load dataset
#     train_dataset = CustomDogDataset(root_dir, transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

#     # Instantiate the ResNet model
#     feature_extractor = ResNetFeatureExtractor().to(device)
#     model = DogSimilarityClassifier(feature_extractor).to(device)

#     # Define loss function and optimizer
#     criterion = TripletLoss(margin=1.0)
#     optimizer = optim.Adam(model.parameters(), lr=0.0001)

#     # Mixed precision training
#     scaler = torch.cuda.amp.GradScaler()

#     # Record start time
#     start_time = time.time()

#     # Training loop
#     num_epochs = 1
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for triplet in generate_triplets(train_dataset, batch_size=32):
#             anchor, positive, negative = triplet
#             anchor, positive, negative = anchor.to(device, non_blocking=True), positive.to(device, non_blocking=True), negative.to(device, non_blocking=True)
#             optimizer.zero_grad()
#             with torch.cuda.amp.autocast():
#                 anchor_out = model(anchor)
#                 positive_out = model(positive)
#                 negative_out = model(negative)
#                 loss = criterion(anchor_out, positive_out, negative_out)
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             running_loss += loss.item() * anchor.size(0)
#         epoch_loss = running_loss / len(train_loader.dataset)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

#     # Record end time
#     end_time = time.time()
#     total_time = end_time - start_time
#     print(f"Total training time: {total_time:.2f} seconds")

#     # Save the trained model
#     os.makedirs("models", exist_ok=True)
#     torch.save(model.state_dict(), "models/dog_similarity_resnet50_triplet.pth")
#     torch.save(feature_extractor.state_dict(), "models/resnet_feature_extractor.pth")

#     print("Training finished and model saved.")

##===========================================================================================================================================

# import torch
# import os
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image, UnidentifiedImageError
# import random
# from run import DogSimilarityClassifier, ResNetFeatureExtractor
# import time

# # Define device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Define transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Define custom dataset class for flat directory
# class CustomDogDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = []

#         for file in os.listdir(root_dir):
#             if file.lower().endswith((".jpg", ".jpeg", ".png")):
#                 self.image_paths.append(os.path.join(root_dir, file))

#         if not self.image_paths:
#             raise ValueError(f"No images found in directory: {root_dir}")

#         # Filter out corrupted images
#         self.image_paths = self.filter_corrupted_images(self.image_paths)
#         self.labels = [0] * len(self.image_paths)

#     def filter_corrupted_images(self, image_paths):
#         valid_paths = []
#         for img_path in image_paths:
#             try:
#                 with Image.open(img_path) as img:
#                     img.verify()  # Verify that the file is a readable image
#                 valid_paths.append(img_path)
#             except (OSError, UnidentifiedImageError):
#                 print(f"Skipping corrupted image: {img_path}")
#         return valid_paths

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         try:
#             image = Image.open(img_path).convert("RGB")
#         except (OSError, UnidentifiedImageError) as e:
#             print(f"Error loading image {img_path}: {e}")
#             return None  # Return None if image is corrupted

#         if self.transform:
#             image = self.transform(image)
        
#         label = self.labels[idx]
#         return image, label

# # Define Triplet Loss
# class TripletLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(TripletLoss, self).__init__()
#         self.margin = margin

#     def forward(self, anchor, positive, negative):
#         positive_distance = torch.nn.functional.pairwise_distance(anchor, positive)
#         negative_distance = torch.nn.functional.pairwise_distance(anchor, negative)
#         loss = torch.nn.functional.relu(positive_distance - negative_distance + self.margin)
#         return loss.mean()

# # Create triplets from your data
# def generate_triplets(dataset, batch_size=32):
#     triplets = []

#     while len(triplets) < len(dataset):
#         indices = random.sample(range(len(dataset)), 3)
#         anchor, positive, negative = dataset[indices[0]], dataset[indices[1]], dataset[indices[2]]
        
#         if None not in (anchor, positive, negative):
#             anchor_img, _ = anchor
#             positive_img, _ = positive
#             negative_img, _ = negative
#             triplets.append((anchor_img, positive_img, negative_img))

#     return DataLoader(triplets, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# if __name__ == '__main__':
#     # Define the root directory containing dog images
#     root_dir = r"E:\gpu\Sphase\samedog6\data\football_database"

#     # Load dataset
#     train_dataset = CustomDogDataset(root_dir, transform=transform)
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

#     # Instantiate the ResNet model
#     feature_extractor = ResNetFeatureExtractor().to(device)
#     model = DogSimilarityClassifier(feature_extractor).to(device)

#     # Define loss function and optimizer
#     criterion = TripletLoss(margin=1.0)
#     optimizer = optim.Adam(model.parameters(), lr=0.0001)

#     # Mixed precision training
#     scaler = torch.cuda.amp.GradScaler()

#     # Record start time
#     start_time = time.time()

#     # Training loop
#     num_epochs = 1
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for triplet in generate_triplets(train_dataset, batch_size=32):
#             anchor, positive, negative = triplet
#             anchor, positive, negative = anchor.to(device, non_blocking=True), positive.to(device, non_blocking=True), negative.to(device, non_blocking=True)
#             optimizer.zero_grad()
#             with torch.cuda.amp.autocast():
#                 anchor_out = model(anchor)
#                 positive_out = model(positive)
#                 negative_out = model(negative)
#                 loss = criterion(anchor_out, positive_out, negative_out)
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             running_loss += loss.item() * anchor.size(0)
#         epoch_loss = running_loss / len(train_loader.dataset)
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

#     # Record end time
#     end_time = time.time()
#     total_time = end_time - start_time
#     print(f"Total training time: {total_time:.2f} seconds")

#     # Save the trained model
#     os.makedirs("models", exist_ok=True)
#     torch.save(model.state_dict(), "models/dog_similarity_resnet50_triplet.pth")
#     torch.save(feature_extractor.state_dict(), "models/resnet_feature_extractor.pth")
