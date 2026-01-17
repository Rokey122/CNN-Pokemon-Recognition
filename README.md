# CNN-Pokemon-Recognition
Repo for my ML assignment for the EESTEC Zagreb AI dev team.

This was a learning assignment on how to use PyTorch to develop an ML model. The assignment was to create and train a model that will recognize what Pokemon is on an input image. The model is trained on 150 different Pokemon.

To use it yourself, first set up PyTorch on your PC. 
You can tweak the hyperparameters in the hyperparams.py, change the model's architecture, optimizer, loss function... 
Train the model simply by running 
```python
python train.py
```
During training you can see the train and test loss and accuracy after every epoch, and after training the model will save itself under model.pt.
You can test the model by downloading a picture, running test.py and inputing the image file name, after which the model will make its prediction.
