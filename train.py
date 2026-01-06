"""Imports"""
import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

import zipfile
from pathlib import Path

import PIL
from PIL import Image

from tqdm import tqdm
from timeit import default_timer as timer 

import matplotlib.pyplot as plt

"""Import helper functions from mrdbourke's Github repo"""
import requests
if not Path("helper.py").is_file():
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
  with open("helper.py", "wb") as f:
    f.write(request.content)
import helper




"""Hyperparameters"""
IMAGE_SIZE = 224
BATCH_SIZE = 32
HIDDEN_UNITS = 64
HIDDEN_UNITS_MULTIPLICATION = 2
CONV_LAYERS = 5
CONVS_IN_LAYER = 2
CLASSIFICATION_HIDDEN_UNITS = 64
SEED = 42
LR = 0.0002
EPOCHS = 100
DROPOUT_LINEAR = 0.4
TRIVIAL_AUGMENT_BINS = 30

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)



"""The model"""
class PokemonRecognition(nn.Module):

  class CNN_Block(nn.Module):
    def __init__(self, in_channels, hidden_units):
      super().__init__()
      self.conv_block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=hidden_units,
                  kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(num_features=hidden_units),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                  kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(num_features=hidden_units),
        nn.MaxPool2d(kernel_size=2)
      )

    def forward(self, x):
      return self.conv_block(x)


  
  def __init__(self, in_channels=3, cnn_hidden_units=50, classification_hidden_units=50,
               out_units=0):
    super().__init__()
    self.cnn = nn.Sequential(
        self.CNN_Block(in_channels=in_channels, hidden_units=cnn_hidden_units),
        self.CNN_Block(in_channels=cnn_hidden_units,
                       hidden_units=cnn_hidden_units*HIDDEN_UNITS_MULTIPLICATION),
        self.CNN_Block(in_channels=cnn_hidden_units*HIDDEN_UNITS_MULTIPLICATION,
                       hidden_units=cnn_hidden_units*HIDDEN_UNITS_MULTIPLICATION**2),
        self.CNN_Block(in_channels=cnn_hidden_units*HIDDEN_UNITS_MULTIPLICATION**2,
                       hidden_units=cnn_hidden_units*HIDDEN_UNITS_MULTIPLICATION**3),
        self.CNN_Block(in_channels=cnn_hidden_units*HIDDEN_UNITS_MULTIPLICATION**3,
                       hidden_units=cnn_hidden_units*HIDDEN_UNITS_MULTIPLICATION**4)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=int(cnn_hidden_units*(HIDDEN_UNITS_MULTIPLICATION**(CONV_LAYERS-1))*((IMAGE_SIZE/(2**(CONV_LAYERS)))**2)),
                  out_features=classification_hidden_units),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=classification_hidden_units),
        nn.Dropout(p=DROPOUT_LINEAR),
        nn.Linear(in_features=classification_hidden_units, out_features=out_units)
    )
    

  def forward(self, x):
    x = self.cnn(x)
    x = self.classifier(x)
    return x



def main():

  """Device agnostic code"""
  if torch.cuda.is_available():
    device = "cuda"
  else:
    device = "cpu"



  """Unpacking and loading data"""
  if not Path("pokemon_data").is_dir():
    with zipfile.ZipFile("pokemon_data.zip", "r") as zip_ref:
      zip_ref.extractall()

  data_path = "pokemon_data"
  train_dir = Path(data_path) / "train"
  test_dir = Path(data_path) / "test"



  image_transform = v2.Compose([
      v2.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
      v2.TrivialAugmentWide(num_magnitude_bins=TRIVIAL_AUGMENT_BINS),
      v2.ToTensor()
  ])

  image_transform_test = v2.Compose([
      v2.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
      v2.ToTensor()
  ])

  train_data = datasets.ImageFolder(train_dir, transform=image_transform, target_transform=None)
  test_data = datasets.ImageFolder(test_dir, transform=image_transform_test, target_transform=None)

  train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
  test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)



  model = PokemonRecognition(in_channels=3, cnn_hidden_units=HIDDEN_UNITS,
                            classification_hidden_units=CLASSIFICATION_HIDDEN_UNITS,
                            out_units=len(train_dataloader.dataset.classes)
                              )
  model=model.to(device)



  """Defining the cost function and the optimizer"""
  cost_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
  scaler = torch.amp.GradScaler("cuda")


  """Training the model"""
  start = timer()
  for epoch in tqdm(range(EPOCHS)):
    print("\nEpoch: ", epoch)
    train_loss, test_loss = 0, 0
    train_acc, test_acc = 0, 0
    model.train()
    for batch, (x, y) in enumerate(train_dataloader):
      if batch % 50 == 0:
        print("   Batch: ", batch)
      x, y = x.to(device), y.to(device)
      
      with torch.amp.autocast("cuda"):
        preds = model(x)
        loss = cost_fn(preds, y)

      train_loss += loss.item()
      train_acc += helper.accuracy_fn(y_true=y, y_pred=preds.argmax(dim=1))

      optimizer.zero_grad()
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

    print("Train loss: ", train_loss/len(train_dataloader))
    print("Train acc: ", train_acc/len(train_dataloader))

    model.eval()
    with torch.inference_mode():
      for batch, (x, y) in enumerate(test_dataloader):
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda"):
          preds = model(x)
          loss = cost_fn(preds, y)

        test_loss += loss.item()
        test_acc += helper.accuracy_fn(y_true=y, y_pred=preds.argmax(dim=1))
    
    print("Test loss: ", test_loss/len(test_dataloader))
    print("Test acc: ", test_acc/len(test_dataloader))

  end = timer()
  print(f"Total training time: {end-start}")

  torch.save(model, Path("model.pt"))

if __name__ == "__main__":
  main()