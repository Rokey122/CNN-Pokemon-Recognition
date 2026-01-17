"""Imports"""
import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

import zipfile
from pathlib import Path

from tqdm import tqdm

from hyperparams import *
from model import *

"""Import helper functions from mrdbourke's Github repo"""
import requests
if not Path("helper.py").is_file():
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
  with open("helper.py", "wb") as f:
    f.write(request.content)
import helper




def main():

  """Setting up seeds for reproduction"""
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)



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

  train_data = datasets.ImageFolder(train_dir, transform=PokemonRecognition.image_transform_train, target_transform=None)
  test_data = datasets.ImageFolder(test_dir, transform=PokemonRecognition.image_transform_test, target_transform=None)

  train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
  test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)



  """Defining the model"""
  model = PokemonRecognition(in_channels=3, cnn_hidden_units=HIDDEN_UNITS,
                            classification_hidden_units=CLASSIFICATION_HIDDEN_UNITS,
                            classes=train_data.classes) 
  model=model.to(device)



  """Defining the cost function and the optimizer"""
  cost_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR)
  scaler = torch.amp.GradScaler("cuda")



  """Training the model"""
  for epoch in tqdm(range(EPOCHS)):

    train_loss, test_loss = 0, 0
    train_acc, test_acc = 0, 0

    model.train()

    for batch, (x, y) in enumerate(train_dataloader):

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

  torch.save(model, Path("model.pt"))



if __name__ == "__main__":
  main()