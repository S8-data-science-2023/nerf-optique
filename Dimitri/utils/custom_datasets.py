from torch.utils.data import DataLoader, Dataset
import os
import cv2

class CustomImageDataset(Dataset):
  def __init__(self, img_folder, transform):
    self.transform=transform
    self.img_folder=img_folder
    self.image_names = list(os.listdir(img_folder))
     
  def __len__(self):
    return len(self.image_names)
 
  def __getitem__(self, index):
     
    image=cv2.imread(self.img_folder + self.image_names[index])
    # image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
 
    image=self.transform(image).unsqueeze(0)
    # targets=self.labels[index]
     
    # sample = {'image': image,'labels':targets}
    sample = {'image': image}
 
    return sample