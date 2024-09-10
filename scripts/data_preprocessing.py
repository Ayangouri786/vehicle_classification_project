import os
from PIL import Image
from torchvision import transforms


def preprocess_image(image_path, output_path, transform):
    image = Image.open(image_path)
    image = transform(image)
    image = transforms.ToPILImage()(image)  # Convert back to PIL Image for saving
    image.save(output_path)  # Save the preprocessed image


def preprocess_dataset(input_dir, output_dir, transform):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for category in os.listdir(input_dir):
        category_path = os.path.join(input_dir, category)
        output_category_path = os.path.join(output_dir, category)
        if not os.path.exists(output_category_path):
            os.makedirs(output_category_path)

        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            output_path = os.path.join(output_category_path, image_name)
            preprocess_image(image_path, output_path, transform)


def main():
    train_input_dir = 'vehicle_dataset/train'
    train_output_dir = 'processed_data/train'
    val_input_dir = 'vehicle_dataset/val'
    val_output_dir = 'processed_data/val'

    transform = transforms.Compose([
        transforms.RandomResizedCrop(180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    preprocess_dataset(train_input_dir, train_output_dir, transform)
    preprocess_dataset(val_input_dir, val_output_dir, transform)


if __name__ == "__main__":
    main()
