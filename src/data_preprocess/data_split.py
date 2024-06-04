import os
import pathlib
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

transform = transforms.Compose(
    [
        # transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]
)

# Load the dataset
data_dir = pathlib.Path("../../kag2")
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# split the dataset into train and test sets
train_size = int(0.8 * len(full_dataset))
test_size = int(0.2 * len(full_dataset))

# train_size+test_size

train_dataset, test_dataset = random_split(full_dataset, [train_size + 1, test_size])

class_names = full_dataset.classes

# Create directories for train and test datasets
train_dir = "../../data/train"
test_dir = "../../data/test"

for class_name in class_names:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)


# function to save images
def save_dataset(data_loader, save_dir):
    for b, (images, labels) in enumerate(data_loader):
        for img_idx in range(images.size(0)):
            img = images[img_idx]
            label = labels[img_idx]
            class_name = class_names[label]

            # img_pil = transforms.ToPILImage()(img)
            save_path = os.path.join(save_dir, class_name, f"{b * 1 + img_idx}.jpeg")
            save_image(img, save_path)


# create dataloaders
# batch_size = 32
train_dataloader = DataLoader(train_dataset, shuffle=True)
test_dataloader = DataLoader(test_dataset, shuffle=True)

save_dataset(train_dataloader, train_dir)
save_dataset(test_dataloader, test_dir)
