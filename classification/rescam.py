import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch
from classification.resnet_pytorch import SE_Block
from classification.cbam import SpatialGate

class ResNetCam(nn.Module):
    def __init__(self,model,synset_loc='../../../datasets/ILSVRC/LOC_synset_mapping.txt',dataset='ImageNet-LT',transformed=True):
        super(ResNetCam, self).__init__()
        self.transformed = transformed
        # get the model
        self.backbone = nn.Sequential(*list(model.children())[:-2])
        
        # get the avg pool of the features stem
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # get the classifier of the model
        self.classifier = model.fc
        
        # placeholder for the gradients
        self.gradients = None
        self.dataset = dataset
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        
        if self.dataset =='ImageNet-LT':
            num_classes=1000
            with open(synset_loc,'r') as file:
                names= file.readlines()
                self.names = [' '.join(n.split(' ')[1:]).rstrip() for n in names]
            with open('../../../datasets/ImageNet-LT/ImageNet_LT_train.txt') as f:
                targets = [int(line.split()[1]) for line in f]
        elif self.dataset =='inat18':
            num_classes=8142
            with open(synset_loc,'r') as file:
                names= file.readlines()
                self.names = [n.rstrip() for n in names]
            with open('../../../datasets/train_val2018/iNaturalist18_train.txt') as f:
                targets = [int(line.split()[1]) for line in f]

        cls_num_list_old = [np.sum(np.array(targets) == i) for i in range(num_classes)]

        # generate class_map: class index sort by num (descending)
        sorted_classes = np.argsort(-np.array(cls_num_list_old))
        class_map = [0 for i in range(num_classes)]
        for i in range(num_classes):
            class_map[sorted_classes[i]] = i
        self.class_map = class_map
        
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.backbone(x)
        
        # register the hook
        h = x.register_hook(self.activations_hook)
        
        # apply the remaining pooling
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.backbone(x)
    
    def show_cat(self,img,index):
        if self.transformed is False:
            img = self.transforms(img).cuda()
        pred = self.forward(img)
        category =  pred.argmax(axis=1)[index].item()
        if self.class_map is not None:
            if self.dataset =='ImageNet-LT':
                if category >= 864:
                    indicator ='R'
                elif (category >= 385) & (category < 864):
                    indicator ='C'
                else:
                    indicator = 'F'
            elif self.dataset =='inat18':
                if category >= 864:
                    indicator ='R'
                elif (category >= 385) & (category < 864):
                    indicator ='C'
                else:
                    indicator = 'F'
            return indicator+': '+ self.names[self.class_map.index(category)]
        else:
            return self.names[category]
        
           
    def show_activation(self,image):
        if self.transformed is False:
            img = self.transforms(image).unsqueeze(0).cuda()
        else:
            img=image.unsqueeze(0)
        pred = self.forward(img)
        
        max_pred = pred.argmax().item()
        pred[:, max_pred].backward(retain_graph=True)
        
        gradients = self.get_activations_gradient()
        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = self.get_activations(img).detach()
        # weight the channels by corresponding gradients
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap.cpu(), 0)

        # normalize the heatmap
        heatmap = heatmap.float()
        heatmap /= torch.max(heatmap)
        try:
            numpy_image = np.asarray(image)
        except TypeError:
            numpy_image = np.asarray(image.cpu())

            numpy_image=numpy_image.transpose(1, 2, 0)
            numpy_image *= np.array(self.transforms.transforms[-1].std)
            numpy_image += np.array(self.transforms.transforms[-1].mean)
            numpy_image = numpy_image*255

        heatmap = cv2.resize(heatmap.numpy(), (numpy_image.shape[1], numpy_image.shape[0]))
        

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + numpy_image
        im2 = np.uint16(superimposed_img)[:,:,::-1]
        
        return im2,
        
    



def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

class SERecorder(nn.Module):
    def __init__(self, model, device = None,part='excitation'):
        super().__init__()
        self.model = model

        self.data = None
        self.recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device
        self.part = part

    def _hook(self, _, input, output):
        self.recordings.append(output.clone().detach())

    def _register_hook(self):
        modules = find_modules(self.model, SE_Block)
        for module in modules:
            if self.part =='excitation':
                handle = module.excitation.register_forward_hook(self._hook)
            elif self.part =='gap':
                handle = module.squeeze.register_forward_hook(self._hook)
            elif self.part =='uniact':
                handle = module.uniact.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.model

    def clear(self):
        self.recordings.clear()

    def record(self, attn):
        recording = attn.clone().detach()
        self.recordings.append(recording)

    def forward(self, img):
        assert not self.ejected, 'recorder has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()

        pred = self.model(img)

        # move all recordings to one device before stacking
        target_device = self.device if self.device is not None else img.device
        recordings = tuple(map(lambda t: t.to(target_device), self.recordings)) 
        attns = recordings if len(recordings) > 0 else None
        return pred, attns
        
        
        
class CBAMRecorder(nn.Module):
    def __init__(self, model, device = None):
        super().__init__()
        self.model = model

        self.data = None
        self.recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

    def _hook(self, _, input, output):
        self.recordings.append(output.clone().detach())

    def _register_hook(self):
        modules = find_modules(self.model, SpatialGate)
        for module in modules:
            handle = module.spatial.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.model

    def clear(self):
        self.recordings.clear()

    def record(self, attn):
        recording = attn.clone().detach()
        self.recordings.append(recording)

    def forward(self, img):
        assert not self.ejected, 'recorder has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()

        pred = self.model(img)

        # move all recordings to one device before stacking
        target_device = self.device if self.device is not None else img.device
        recordings = tuple(map(lambda t: t.to(target_device), self.recordings)) 
        attns = recordings if len(recordings) > 0 else None
        return pred, attns