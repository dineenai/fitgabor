# Fit Gabor

If you have a neuron model and you want to find the gabor filter that maximizes the activity of your model neuron, this tool might come handy.

## Example

### 1. We need a neuron model

Let's create a simple linear-nonlinear neuron model with a known receptive field:
``` python
import torch
from torch import nn
from torch.nn import functional as F

class Neuron(nn.Module):
    def __init__(self, rf):
        super().__init__()
        self.rf = torch.tensor(rf.astype(np.float32))
        
    def forward(self, x):
        return F.elu((x * self.rf).sum()) + 1
```
Here is the **ground truth** RF:
<p align="center">
  <img width="300" height="300" src="figures/groundtruth1.png">
</p>

### 2. Initialize and train the gabor generator

``` python
gabor_gen = GaborGenerator(image_size=(64, 64))
gabor_gen, _ = trainer_fn(gabor_gen, neuron,
                          epochs=20000,
                          sigma_reg_lambda=7.25)
```

You can generate the learned gabor simply by calling the model:
``` python
learned_gabor = gabor_gen().data.numpy()
```

And here is the learning evolution:
<p align="center">
  <img width="230" height="230" src="figures/evolution1.gif">
</p>

### 3. Tried it and didn't work?

Let me know, please! I would love to know why it did not work and help fix it :construction_worker:

---
## More examples

Ground turth             |  Learned
:-------------------------:|:-------------------------:
![](figures/groundtruth0.png)  |  ![](figures/evolution0.gif)
![](figures/groundtruth2.png)  |  ![](figures/evolution2.gif)
![](figures/groundtruth3.png)  |  ![](figures/evolution3.gif)
![](figures/groundtruth4.png)  |  ![](figures/evolution4.gif)
