import torch
from torchvision import  models

from DataInformation import dataloaders
#!/usr/bin/env python

fcModels= ["resnet34", 'inception_v3']

class CheckPointManupulationSystem:

    def load_checkpoint(self,filepath):
        
        checkpoint = torch.load(filepath)
        model = getattr(models, checkpoint['arch'])(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        
        if checkpoint["arch"] in fcModels : model.fc = checkpoint["classifier"]
        else:   model.classifier = checkpoint['classifier']
        
        model.state_dict = checkpoint['state_dict']
        model.class_to_idx = checkpoint['class_to_idx']
        return model


    def save_checkpoint(self, args, model, optimizer):
        
        checkpoint = {'arch':args.arch,
                'optimizer':optimizer,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx,
                'hidden_units':args.hidden_units,
                'learning_rate':args.learning_rate,
                'epochs':args.epochs,
                'classifier':model.fc if args.arch in fcModels  else  model.classifier}
        print('Saving checkpoint.')
        torch.save(checkpoint, args.checkpoint)

