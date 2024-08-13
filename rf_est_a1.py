# TO DO: Output to txt file - also save images as sanity check
# Loopover images etc.....


#/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/conv1_filters/supRN50_conv1_21_g0_60e_e60_np_arr

#eg. supRN50_conv1_21_g0_60e_e60_filter_np_arr_6.npy

# supRN50_conv1_21_g4_30e_g0_30e_e60_np_arr
# supRN50_conv1_21_g0_30e_g4_30e_e60_np_arr
# supRN50_conv1_21_g4_60e_e60_np_arr

import numpy as np
import matplotlib.pyplot as plt
from fitgabor import GaborGenerator, trainer_fn
from fitgabor.utils import gabor_fn

import torch
from torch import nn
from torch.nn import functional as F

class Neuron(nn.Module):
    def __init__(self, rf):
        super().__init__()
        h, w = rf.shape
        self.rf = torch.tensor(rf.reshape(1, 1, h, w).astype(np.float32))
        
    def forward(self, x):
        return F.elu((x * self.rf).sum()) + 1


# Upload filter - try single file first - then iterate through folder
# - ideally iterate through subdirectories ending in np_arr too
path_to_rf = "/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/visualizing_filters/conv1_filters/supRN50_conv1_21_g0_60e_e60_np_arr/supRN50_conv1_21_g0_60e_e60_filter_np_arr_6.npy"
rf = np.load(path_to_rf)

# Make image greyscale - is this the best way to do it?
# https://stackoverflow.com/questions/14365029/numpy-3d-image-array-to-2d
gray = rf[0,:,:]

# Will not show on server? Export?
# Q: Why does image not show in black and white
fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
ax.imshow(gray);
ax.set(xticks=[], yticks=[]);

rf = gray


neuron = Neuron(rf)

# Create a gabor generator
_, _, h, w = neuron.rf.shape
# WHy 420?
torch.manual_seed(420)
gabor_gen = GaborGenerator(image_size=(h, w))

# Gabor vs. true RF before training
learned_rf = gabor_gen().squeeze().cpu().data.numpy()
true_rf = neuron.rf.squeeze().cpu().data.numpy()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=100)
ax1.imshow(learned_rf);
ax1.set(xticks=[], yticks=[], title="Learned gabor")

ax2.imshow(true_rf);
ax2.set(xticks=[], yticks=[], title="True RF");


# Train the gabor generator to maximizes the model output
from torch import optim #Move to top of file
gabor_gen, evolved_rfs = trainer_fn(gabor_gen, neuron,save_rf_every_n_epoch=100)

# Learning evolution of the gabor generator
print(len(evolved_rfs))

n_rows = 4
n_cols = (len(evolved_rfs) + n_rows - 1) // n_rows

fig, axes = plt.subplots(n_rows, n_cols, dpi=100, figsize=(20, 12))

for ind, ax in enumerate(axes.flat):
    if ind < len(evolved_rfs):
        ax.imshow(evolved_rfs[ind])
        ax.set(xticks=[], yticks=[])
    else:
        ax.axis('off')

# Gabor vs. true RF after training
learned_rf = gabor_gen().squeeze().cpu().data.numpy()
true_rf = neuron.rf.squeeze().cpu().data.numpy()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3), dpi=100)

# ax2.imshow(true_rf);
# ax1.set(xticks=[], yticks=[], title="True RF")

# ax1.imshow(learned_rf);
# ax2.set(xticks=[], yticks=[], title="Learned gabor");

ax1.imshow(true_rf);
ax1.set(xticks=[], yticks=[], title="True RF")

ax2.imshow(learned_rf);
ax2.set(xticks=[], yticks=[], title="Learned gabor");


#What are the parameters?

      

a=gabor_gen.center.detach().numpy()
print(f"{a},{np.array(gabor_gen.image_size)},{gabor_gen.sigma.detach().numpy()},{np.array(gabor_gen.theta.detach())}")
# # Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.


print(f"{a},{np.array(gabor_gen.image_size)}")
print(f"{a},{gabor_gen.sigma.detach().numpy()}")
print(f"{a},{np.array(gabor_gen.theta.detach().numpy())}")

# TO DO: Output to txt file - also save images as sanity check

# 13
# [1.4066708 1.5699201],[21 21],[3.5319695],[3.1415927]
# [1.4066708 1.5699201],[21 21]
# [1.4066708 1.5699201],[3.5319695]
# [1.4066708 1.5699201],[3.1415927]



# Original Notebook Code - FIX
# a=gabor_gen.center.detach().numpy()
# print(f"{a},{np.array(gabor_gen.image_size)},{gabor_gen.sigma.detach().numpy()},{np.array(gabor_gen.theta)}")

















# Image 0
# [1.7033962 1.7744944],[21 21]
# [1.7033962 1.7744944],[3.7531383]
# [1.7033962 1.7744944],[1.4981645]

# What do these two numbers mean? 
# [1.7033962 1.7744944]


# Image 3 - NB NOTE THAT GABOR IS NOT A GOOD FIT IN THIS CASE - IMPOSE A THRESHOLD! - GENERATE A METRIC
# [0.67705107 0.45202926],[21 21],[3.],[1.5081306]
# [0.67705107 0.45202926],[21 21]
# [0.67705107 0.45202926],[3.]
# [0.67705107 0.45202926],[1.5081306]


# How can I get a measure of goodness of fit?

# a=gabor_gen.center.detach().numpy()

# print(f"{a},{np.array(gabor_gen.image_size)},{gabor_gen.sigma.tensor.detach().numpy()},{np.array(gabor_gen.theta)}")




