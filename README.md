# SIM-pytorch
Official code for Self Identity Mapping.

## Quick Example
**Note:**
When using the @init_proj decorator, you must provide the names of the module’s input and output dimensions (e.g., in_channels and out_channels). These names must correspond to attributes of the target block itself (for example, self.in_channels and self.out_channels inside the class). SIM uses these attributes to determine the correct shape of the reconstructor network. If these attributes are missing or not defined in the block, the initialization will fail.
```python
import torch
import torch.nn as nn
from sim_loss import init_proj, forward_with_sim_cache, compute_loss_sim, LossSimTracker

# Define a block with SIM hooks
class Block(nn.Module):
    # Step 1
    @init_proj(input_dim_name="in_channels", output_dim_name="out_channels")
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.f = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.in_channels = in_channels
        self.out_channels = out_channels
    # Step 2
    @forward_with_sim_cache()
    def forward(self, x):
        return self.f(x)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*[Block(3, 3) for _ in range(3)])

    def forward(self, x):
        return self.backbone(x)

model = Model().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop example
for _ in range(10):
    x = torch.randn(2, 3, 224, 224).cuda()
    gt = torch.rand_like(x, requires_grad=True)
    
    optimizer.zero_grad()
    
    y = model(x)
    
    loss = (y - gt).mean()

    # Step 3
    loss += compute_loss_sim(model)

    loss.backward()

    optimizer.step()

    print(f"Loss: {loss.item():.6f}")
```

Cached Tracker Usage

The cached version avoids re-scanning your model at every step:
```python
from sim_loss import init_proj, forward_with_sim_cache, compute_loss_sim, LossSimTracker

model = Model().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

loss_sim_tracker = LossSimTracker(model)

# Training loop example
for _ in range(10):
    x = torch.randn(2, 3, 224, 224).cuda()
    gt = torch.rand_like(x, requires_grad=True)
    
    optimizer.zero_grad()
    
    y = model(x)
    
    loss = (y - gt).mean()

    # Step 3
    loss += loss_sim_tracker.compute_loss_sim()

    loss.backward()

    optimizer.step()

    print(f"Loss: {loss.item():.6f}")
```

## TODO

- [ ] Release a preprint of the paper.
- [ ] Release pipeline code for various tasks, including image classification and domain generalization, etc.
