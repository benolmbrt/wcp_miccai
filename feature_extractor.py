import torch
import torch.nn as nn
from tqdm import tqdm
from ...features.feature_utils import set_forward_hooks, remove_forward_hooks, filter_activations
import copy

"""
Returns the average activation of each convolution feature map in the net.
"""

def getActivation(activation_dict, name):
    # the hook signature
    def hook(model, input, output):
        activ = output.detach()
        avg_activ = torch.mean(activ, dim=[2, 3, 4])
        activation_dict[name] += [avg_activ.cpu()]

    return hook

class Extractor(nn.Module):

    def __init__(self, layer=None, verbose=False):
        super().__init__()

        self.activation_dict = {}
        self.dtype = torch.float32
        self.means = None
        self.stds = None
        self.layer = layer
        self.verbose = verbose

    def reset(self):
        for key in self.activation_dict:
            self.activation_dict[key] = []

    def predict(self, model, x):

        batch_size = len(x)
        batch_activations = []
        hooks = set_forward_hooks(self.activation_dict, getActivation, model, verbose=False, layer_names=self.layer)
        self.reset()

        for b in range(batch_size):
            x_b = x[b, ...].unsqueeze(0)
            model.prediction_wrapper(x_b)

            filter_activations(self.activation_dict)
            batch_activations += [copy.deepcopy(self.activation_dict)]
            self.reset()  # re-initialize dict

            if model.hparams.patch_training:
                # average activation over patches
                for key in self.activation_dict:
                    patch_activs = batch_activations[b][key]
                    batch_activations[b][key] = torch.mean(patch_activs, dim=0, keepdim=True)

        # remove hooks, otherwise they are still attached to the model, which will eventually yield to a bug if
        # predict is called multiple times
        remove_forward_hooks(hooks, verbose=self.verbose)
        return batch_activations
