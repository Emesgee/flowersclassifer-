import torch
import os
from torchvision import transforms, datasets, models
from collections import OrderedDict
from TrainingSystem import TrainingSystem
import torch.nn as nn
from  DataInformation import dataloaders, data_dir, data_transforms, dataset_sizes, image_datasets
import copy
from CheckPointManupulationSystem import CheckPointManupulationSystem

#from ParseArguments import ParsingArguments


class Model(TrainingSystem):

    def __init__(self):
        ''

    def paramaterFreezing(self, model):
        for param in model.parameters():
            param.requires_grad = False


    def trainTheModel(self, device, args, model):

        self.initiateDataStructure_model(model)

        self.initiateClassifer(args, model)
        
        #defining the optimizer, criteriom and the scheduler
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)
        model = model.to(device)
        #model = train_model(model, optimizer, criterion, scheduler, image_datasets, args.gpu, args.epochs)
        model = self.train_model(device,model, self.optimizer, criterion, scheduler,  args.gpu, args.epochs)
        model.class_to_idx = image_datasets['train'].class_to_idx

        chcPoint=CheckPointManupulationSystem()
        chcPoint.save_checkpoint(args,model, self.optimizer)

    def run(self, device, args, model):
            self.paramaterFreezing(model);
            self.trainTheModel(device,args, model);




class AlexNet (Model):
    def initiateDataStructure_model(self,  model):
        self.in_ft = model.classifier[1].in_features
        self.out_ft = 102 


       
    def initiateClassifer(self, args, model):

        self.classifier = torch.nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(self.in_ft, args.hidden_units)),
                    ('relu', nn.ReLU()),
                    ('fc2', nn.Linear(args.hidden_units, self.out_ft)),
                    ('output', nn.LogSoftmax(dim=1))
            ]))
        model.classifier = self.classifier
        self.optimizer = torch.optim.SGD(model.classifier.parameters(), lr=args.learning_rate, momentum=0.9)

class VGGNet (Model):
    def initiateDataStructure_model(self, model):
        self.in_ft = model.classifier[0].in_features
        self.out_ft = 102 
       
    def initiateClassifer(self, args, model):
        self.classifier = torch.nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(self.in_ft, 512)),
                    ('relu', nn.ReLU()),
                    ('drpot', nn.Dropout(p=0.5)),
                    ('hidden', nn.Linear(512, args.hidden_units)),                       
                    ('fc2', nn.Linear(args.hidden_units, self.out_ft)),
                    ('output', nn.LogSoftmax(dim=1))
            ]))

        model.classifier = self.classifier
        self.optimizer = torch.optim.SGD(model.classifier.parameters(), lr=args.learning_rate, momentum=0.9)

class DenseNet (Model):
    def initiateDataStructure_model(self, model):
        self.in_ft = model.classifier.in_features
        self.out_ft = 102 
       
    def initiateClassifer(self, args, model):
        self.classifier = torch.nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(self.in_ft, 512)),
                    ('relu', nn.ReLU()),
                    ('drpot', nn.Dropout(p=0.5)),
                    ('hidden', nn.Linear(512, args.hidden_units)),                       
                    ('fc2', nn.Linear(args.hidden_units, self.out_ft)),
                    ('output', nn.LogSoftmax(dim=1))
            ]))

        model.classifier = self.classifier
        self.optimizer = torch.optim.SGD(model.classifier.parameters(), lr=args.learning_rate, momentum=0.9)




class fcModel(Model):
    def initiateDataStructure_model(self, model):
        self.in_ft = model.fc.in_features
        self.out_ft = 102 
       
    def initiateClassifer(self, args, model):
        self.classifier = torch.nn.Sequential(OrderedDict([
                    ('fc1', nn.Linear(self.in_ft, 512)),
                    ('relu', nn.ReLU()),
                    ('drpot', nn.Dropout(p=0.5)),
                    ('hidden', nn.Linear(512, args.hidden_units)),                       
                    ('fc2', nn.Linear(args.hidden_units, self.out_ft)),
                    ('output', nn.LogSoftmax(dim=1))
            ]))

        model.fc = self.classifier
        self.optimizer = torch.optim.SGD(model.fc.parameters(), lr=args.learning_rate, momentum=0.9)



class ResNet (fcModel):
    ''


class Inception (fcModel):
    ''



