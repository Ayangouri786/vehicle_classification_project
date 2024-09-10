import os
from PIL import Image

def analyze_dataset(dataset_path):
    categories = os.listdir(dataset_path)
    for category in categories:
        category_path = os.path.join(dataset_path, category)
        num_images = len(os.listdir(category_path))
        print(f"Category: {category}, Number of images: {num_images}")

def main():
    dataset_path = 'vehicle_dataset/train'
    analyze_dataset(dataset_path)
    # Add code here to analyze and clean the dataset if needed

if __name__ == "__main__":
    main()
