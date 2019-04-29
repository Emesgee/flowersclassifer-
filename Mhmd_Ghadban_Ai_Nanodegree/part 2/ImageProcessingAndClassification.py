1#!/usr/bin/env python
import argparse # For command-line interface
from PIL import Image
import glob, os
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import numpy as np
import torch
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
from torch.autograd import Variable
#import densenet
import argparse # For command-line interface
import json
import torch
import PIL
from PIL import Image
from torchvision import models, transforms

from collections import OrderedDict 


class imageProcessing:

    def process_image(self, image):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
        img = Image.open(image)
        img_tensor = preprocess(img)
        return img_tensor


class imagePredictionSystem:
    
    def __init__ (self):
        print('') 



    def load_categories(self,filename):
        with open(filename) as f:
            category_names = json.load(f)
        return category_names


    def predict(self, image_path, model, device, topk=1):
        imgProcessor=imageProcessing()
        processed_image = imgProcessor.process_image(image_path)
        processed_image.unsqueeze_(0)
        output = model(processed_image.to(device))
        ps = torch.exp(output).data.topk(topk)
        probs, indices = torch.topk(F.softmax(output, dim=1), topk, sorted=True)
        probs = ps[0]
        classes = ps[1]
        # Map classes to indices
        #inverted_class_to_idx = {
        #    model.class_to_idx[k]: k for k in model.class_to_idx}
        #mapped_classes = list()
        #for label in classes.numpy()[0]:
        #    mapped_classes.append(inverted_class_to_idx[label])

        # Return results
        #return probs.numpy()[0], mapped_classes


        idx_to_class = { v:k for k, v in model.class_to_idx.items()}
        return ([prob.item() for prob in probs[0].data], 
                [idx_to_class[ix.item()] for ix in indices[0].data])


