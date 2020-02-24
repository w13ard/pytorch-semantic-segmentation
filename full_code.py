# dll import
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
%matplotlib inline

# google drive connect
from google.colab import drive
drive.mount('/content/gdrive')

# goto dataset dir
# dataset\.
#        \train_data
#        \train_mask
cd gdrive/My\ Drive/NeuralNetwork/FinalProject/dataset

# load consts
DEVICE = torch.device("cuda")
RESCALE_SIZE = 224
TRAIN_DATA_DIR = Path('train_data')
TRAIN_MASKS_DIR = Path('train_masks')

# load images (count=5000) and masks (count=5000)
train_images = sorted( list( TRAIN_DATA_DIR.rglob('*.jpg') ) )
train_masks = sorted( list( TRAIN_MASKS_DIR.rglob('*.gif') ) )

# load pretrained model and move it to CUDA
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True).to(DEVICE)

# freeze model backbone because feature extract used
def set_backbone_parameter_requires_grad(model, val):
        for param in model.backbone.parameters():
            param.requires_grad = val
            
# func
set_backbone_parameter_requires_grad(model, False)

# change output classes in classifier from 21 (resnet) to 2 (carvana classes: 1-car, 2-background) and move to CUDA
model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1)).to(DEVICE)
model.aux_classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1)).to(DEVICE)

# params_to_update would contain params that should be passed to optimizer for gradient update
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        print("\t",name)
        
"""
load dataset
images - path to sorted images
labels - path to sorted masks
mode - train / val

loaded images assigned to PIL; resize to 224x224; norm with /.255; transforms (go from numpy to tensor and 
normalize as pretrained model need
loaded masks resize and assigned to FloatTensor

output from dataset is list of tensor image and tensor mask: [3х224х224 (image), 1х224х224 (mask)]
"""
class CarvanaDataset(Dataset):

    def __init__(self, images, labels, mode):
      super().__init__()
      self.images = images
      self.labels = labels
      self.mode = mode

    def __len__(self):
      return len(self.images)
      
    def load_sample(self, file):
      image = Image.open(file)
      image.load()
      return image

    def prepare_sample(self, image):
      image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
      return np.array(image)

    def __getitem__(self, index):

      if self.mode == 'train':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
        
        x = self.load_sample(self.images[index])
        x = self.prepare_sample(x)
        x = np.array(x / 255, dtype='float32')

        y = self.load_sample(self.labels[index])
        y = self.prepare_sample(y)   

        x = transform(x).to(DEVICE)
        y = torch.FloatTensor(y).to(DEVICE)
        return [x,y.unsqueeze(0)]

      if self.mode == 'val':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
        
        x = self.load_sample(self.images[index])
        x = self.prepare_sample(x)
        x = np.array(x / 255, dtype='float32')

        y = self.load_sample(self.labels[index])
        y = self.prepare_sample(y)   

        x = transform(x).to(DEVICE)
        y = torch.FloatTensor(y).to(DEVICE)
        return [x,y]
        
# slice to split dataset to train (80%) и val (20%)
data_slice = int( (len(train_images)/100 ) * 80)

# split to train and val datasets
train_dataset = CarvanaDataset(train_images[:data_slice], train_masks[:data_slice], mode = 'train')
val_dataset = CarvanaDataset(train_images[data_slice:], train_masks[data_slice:], mode = 'val')

# dataloaders, train_loader load in batchs, val_loader whole
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False)

# train func
def make_train_step(model, loss_fn, optimizer):
    def train_step(x, y):
        model.train()
        yhat = model(x)['out']
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step
    
# hyperparams
learning_rate = 1e-5
n_epochs = 30
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params_to_update, lr=learning_rate)

# train_step assign
train_step = make_train_step(model, loss_fn, optimizer)

# main train and eval cycle
training_losses = []
validation_losses = []

for epoch in range(n_epochs):
    batch_losses = []

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        new_y_batch = y_batch.squeeze_(1).long() # squeeze_(1) https://pytorch.org/docs/stable/nn.html#crossentropyloss
        loss = train_step(x_batch, new_y_batch)
        batch_losses.append(loss)
        print(f"batch losses: {loss:.3f}")

    training_loss = np.mean(batch_losses)
    training_losses.append(training_loss)

    print(f"training_losses: {training_loss:.3f}")

    with torch.no_grad():
        val_losses = []
        training_loader_iter = iter(val_loader)
        for i in range(1):
            x_val, y_val = next(training_loader_iter)
            x_val = x_val.to(DEVICE)
            y_val = y_val.to(DEVICE)
            model.eval()
            yhat = model(x_val.unsqueeze_(0))['out']
            new_y_val = y_val.unsqueeze_(0).long()
            val_loss = loss_fn(yhat, new_y_val).item()
            val_losses.append(val_loss)
        validation_loss = np.mean(val_losses)
        validation_losses.append(validation_loss)
    torch.save(model, f'{validation_loss}.pth')
    print('model saved')
    print(f"[{epoch+1}] Training loss: {training_loss:.3f}\t Validation loss: {validation_loss:.3f}")
    
# Define the helper function
def decode_segmap(image, nc=2):
  label_colors = np.array([(0, 0, 0), #background
                           (254, 254, 254),]) #mask
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
  rgb = np.stack([r, g, b], axis=2)
  return rgb

# helper func for image show
def segment(net, show_orig=True, dev='cuda'):
  if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
  trf = transforms.Compose([transforms.Resize(224), 
                   transforms.ToTensor(),
                   transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                               std = [0.229, 0.224, 0.225])])
  inp = trf(img).unsqueeze(0).to(dev)
  out = net.to(dev)(inp)['out']
  om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
  rgb = decode_segmap(om)
  plt.imshow(rgb); plt.axis('off'); plt.show()
  
# model eval
nn = model.eval()
img = Image.open('path/to/image.jpg')
segment(nn, img)
