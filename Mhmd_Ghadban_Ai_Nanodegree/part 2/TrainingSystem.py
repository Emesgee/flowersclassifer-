import torch
import os
from torchvision import transforms, datasets, models
from collections import OrderedDict
from  DataInformation import dataloaders, data_dir, data_transforms, dataset_sizes, image_datasets
import copy
from deviceSystem import deviceSystem
from torch.autograd import Variable ###



class TrainingSystem:

    def train_model(self, device, model, optimizer, criterion, scheduler,  gpu, num_epochs=10):
        
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    if deviceSystem:
                        inputs = Variable(inputs.float().cuda())
                        labels = Variable(labels.long().cuda())
                    else:
                        inputs = Variable(inputs)
                        labels = Variable(labels)

                    
                    #Set gradient to zero     
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                print('Phase: {} Loss: {:.4f} Acc: {:.4f}'.format(
                      phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model
