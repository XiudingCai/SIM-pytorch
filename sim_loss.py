import torch
import torch.nn.functional as F
import torch.nn as nn

from einops import rearrange
from functools import wraps
from einops.layers.torch import Reduce

eps = 1.0e-5


def init_proj(input_dim_name, output_dim_name=None, proj_type='InvSS',
              token_sampling_ratio=0.2, channel_sampling_ratio=1., token_sampling_type='v0',
              loss_type='norm_mse', sim_cfg=None):
    def decorator(init_fn):
        @wraps(init_fn)
        def wrapper(self, *args, **kwargs):
            init_fn(self, *args, **kwargs)

            if isinstance(input_dim_name, int):
                input_dim = input_dim_name
            elif isinstance(input_dim_name, str):
                input_dim = getattr(self, input_dim_name)
            else:
                raise TypeError(f"The type of input_dim_name should be str or int, but found {type(input_dim_name)}!")

            if output_dim_name is None:
                output_dim = input_dim
            elif isinstance(output_dim_name, str):
                output_dim = getattr(self, output_dim_name)
            else:
                output_dim = output_dim_name

            self.sim_cfg = {}
            self.proj_type = self.sim_cfg.get('proj_type', proj_type)
            self.loss_type = self.sim_cfg.get('loss_type', loss_type)
            self.loss_beta = self.sim_cfg.get('loss_beta', 5e-3)
            self.token_sampling_ratio = self.sim_cfg.get('token_sampling_ratio', token_sampling_ratio)
            self.channel_sampling_ratio = self.sim_cfg.get('channel_sampling_ratio', channel_sampling_ratio)
            self.token_sampling_type = token_sampling_type

            print(f"{self.proj_type=}, {self.loss_type=}, {self.loss_beta=}, "
                  f"{self.token_sampling_ratio=}, {self.channel_sampling_ratio=}")

            # the real input dim
            K = max(int(channel_sampling_ratio * output_dim), 1)
            K_in = max(int(channel_sampling_ratio * input_dim), 1)
            K_out = max(int(channel_sampling_ratio * output_dim), 1)

            if proj_type == 'InvSS':
                if K_out != K_in:
                    self.pre_proj = nn.Sequential(
                        nn.Linear(K_out, K_in, bias=False),
                    )
                else:
                    self.pre_proj = nn.Identity()
            else:
                self.pre_proj = nn.Identity()
            if proj_type == 'InvSS':
                K = int(K_in ** 0.5)
                self.proj_x = nn.Sequential(
                    nn.Linear(K_in, K * 2, bias=True),
                    nn.ReLU(),
                    nn.Linear(K * 2, K, bias=True),
                    nn.ReLU(),
                    nn.Linear(K, K, bias=True),
                )
                self.proj_y = self.proj_x
                self.predictor = nn.Sequential(
                    nn.Linear(K, K, bias=True),
                )

        return wrapper

    return decorator


def forward_with_sim_cache():
    def decorator(forward_fn):
        @wraps(forward_fn)
        def wrapper(self, x, *args, **kwargs):
            identity = x
            out = forward_fn(self, x, *args, **kwargs)

            y = out

            if self.training:
                ##########################################################################
                # # STEP 1: prepare x and y with the same shape of (B, N, C)
                ##########################################################################
                x = identity.detach()
                # x = (y - identity).detach()
                # if x.dim() == 4:
                #     # For 4D tensor (B, C, H, W): resize both H and W to match out.shape[2:]
                #     if x.shape[2:] != out.shape[2:]:
                #         x = F.interpolate(x, size=out.shape[2:], mode='bilinear', align_corners=False)
                # elif x.dim() == 3:
                #     # For 3D tensor (B, N, C): resize only the second dimension to match out.shape[1]
                #     if x.shape[1] != out.shape[1]:
                #         x = F.interpolate(x.unsqueeze(-1), size=(out.shape[1], 1), mode='bilinear',
                #                           align_corners=False).squeeze(-1)

                if len(x.shape) == 4:
                    x = rearrange(x, 'b c h w -> b (h w) c')
                if len(y.shape) == 4:
                    y = rearrange(y, 'b c h w -> b (h w) c')

                dim_K = min(max(int(self.channel_sampling_ratio * x.shape[-1]), 1), x.shape[-1])
                if dim_K != x.shape[-1]:

                    x = x[..., :dim_K]
                    y = y[..., :dim_K]

                ##########################################################################
                # # STEP 2: patch-level sampling, (B, N, C) -> (B, n, C)
                ##########################################################################
                token_sampling_ratio = self.token_sampling_ratio
                if token_sampling_ratio == 0:
                    return
                if 0 < token_sampling_ratio <= 1:
                    token_sampling_ratio = int(token_sampling_ratio * x.shape[1])

                if self.token_sampling_type == 'v0':
                    patch_id = torch.randperm(x.shape[1], device=x.device)
                    patch_id = patch_id[:int(min(token_sampling_ratio, patch_id.shape[0]))]

                    x = x[:, patch_id]
                    y = y[:, patch_id]
                elif self.token_sampling_type == 'v1':
                    B, N, _ = x.shape
                    # Generate a different random permutation of patch indices for each sample. rand_ids has shape (B, N)
                    rand_ids = torch.argsort(torch.rand(B, N, device=x.device), dim=1)

                    # Select the first subset of indices per sample according to sampling ratio
                    patch_id = rand_ids[:, :int(min(token_sampling_ratio, N))]

                    # Use gather to apply different sampled indices for each sample
                    x = torch.gather(x.clone(), 1, patch_id.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
                    y = torch.gather(y.clone(), 1, patch_id.unsqueeze(-1).expand(-1, -1, y.shape[-1]))
                elif self.token_sampling_type == 'v2':
                    B, N, _ = x.shape

                    # (B, N, C) -> (B, N)
                    scores_mean = x.detach().mean(dim=-1)

                    # Generate random indices for sampling
                    rand_ids = torch.argsort(torch.rand(B, N, device=x.device), dim=1)
                    # print("rand_ids", rand_ids.min(), rand_ids.max())

                    # perform oversampling for each sample, e.g. (B, 3n)
                    rand_ids = rand_ids[:, :int(min(token_sampling_ratio * 3, N))]
                    # print("rand_ids", rand_ids.min(), rand_ids.max())

                    # gather values from y_mean using random indices, (B, 3n)
                    scores_mean = scores_mean.gather(1, rand_ids)

                    top_k_ids = torch.topk(scores_mean, token_sampling_ratio, dim=1).indices

                    # get indices of top-k sampled values
                    patch_id = rand_ids.gather(1, top_k_ids)
                    # print("patch_id", patch_id.min(), patch_id.max())

                    # final sampling, avoiding inplace operation
                    x = torch.gather(x.clone(), 1, patch_id.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
                    y = torch.gather(y.clone(), 1, patch_id.unsqueeze(-1).repeat(1, 1, y.shape[-1]))
                else:
                    raise NotImplementedError

                with torch.no_grad():
                    zx = self.proj_x(x)

                y = self.pre_proj(y)
                zy = self.proj_y(y)

                zy = self.predictor(zy)

                self.loss_sim = compute_sim_loss(zy, zx.detach(), loss_type=self.loss_type, ) * self.loss_beta
            else:
                self.loss_sim = torch.tensor(0., device=x.device)

            return out

        return wrapper

    return decorator


def compute_sim_loss(x_sample, y_sample, loss_type='mse'):
    """
    x, y: B, N, C
    """

    if loss_type == 'cos':
        loss_sim = 2 - 2 * F.cosine_similarity(x_sample, y_sample, dim=-1)

    elif loss_type == 'kl':
        x_sample = F.log_softmax(x_sample, dim=-1)
        y_sample = F.softmax(y_sample, dim=-1)
        loss_sim = F.kl_div(x_sample, y_sample.detach(), reduction='batchmean')

    elif loss_type == 'smooth_l1':
        loss_sim = F.smooth_l1_loss(x_sample, y_sample)

    elif loss_type == 'mse':
        loss_sim = F.mse_loss(x_sample, y_sample)

    elif loss_type == 'norm_mse':
        loss_sim = normalized_euclidean_distance(x_sample, y_sample)

    else:
        raise NotImplementedError

    return loss_sim.mean()


def normalized_euclidean_distance(x, y):
    # Calculate the mean and standard deviation for each sample
    x_mean = x.mean(dim=1, keepdim=True)
    x_std = x.std(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    y_std = y.std(dim=1, keepdim=True)

    # Standardize x and y
    x_normalized = (x - x_mean) / (x_std + 1.e-6)  # Add a small constant to avoid division by zero
    y_normalized = (y - y_mean) / (y_std + 1.e-6)

    # Compute the normalized Euclidean distance
    dist = torch.norm(x_normalized - y_normalized, p=2, dim=-1)  # p=2 indicates L2 norm
    return dist.mean()


def compute_loss_sim(module):
    """
    Compute the average 'loss_sim' of all submodules that have this attribute.

    :param module: torch.nn.Module, the root module to scan
    :return: float, average loss_sim, or 0 if none found
    """

    def collect_loss_sim_modules(mod):
        modules_with_loss = []
        for submodule in mod.children():
            modules_with_loss.extend(collect_loss_sim_modules(submodule))
            if hasattr(submodule, 'loss_sim'):
                modules_with_loss.append(submodule)
        return modules_with_loss

    loss_sim_modules = collect_loss_sim_modules(module)

    if not loss_sim_modules:
        print("[Warning] No modules with 'loss_sim' found.")
        return 0.0

    loss_sum = 0.0
    count = 0
    for submodule in loss_sim_modules:
        loss_value = getattr(submodule, 'loss_sim')
        if not torch.isnan(loss_value):
            loss_sum += loss_value
            count += 1

    return loss_sum / count if count > 0 else 0.0

class LossSimTracker:
    def __init__(self, module):
        self.module = module
        self.loss_sim_modules = None

    def _collect_loss_sim_modules(self, module, parent_name=''):
        modules = []
        for name, submodule in module.named_children():
            full_name = f'{parent_name}.{name}' if parent_name else name
            modules.extend(self._collect_loss_sim_modules(submodule, full_name))
            if hasattr(submodule, 'loss_sim'):
                modules.append(submodule)
        return modules

    def compute_loss_sim(self):
        # cold start
        if self.loss_sim_modules is None:
            self.loss_sim_modules = self._collect_loss_sim_modules(self.module)
            if not self.loss_sim_modules:
                print("[Warning] No modules with 'loss_sim' found on first compute_loss_sim call.")

        loss_sum = 0.0
        count = 0
        for submodule in self.loss_sim_modules or []:
            loss_value = getattr(submodule, 'loss_sim')
            if not torch.isnan(loss_value):
                loss_sum += loss_value
                count += 1
        return loss_sum / count

if __name__ == '__main__':
    class Block(nn.Module):
        # STEP 1
        @init_proj(input_dim_name="in_channels", output_dim_name="out_channels")
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.f = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            self.in_channels = in_channels
            self.out_channels = out_channels

        # STEP 2
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

    for _ in range(10):
        x = torch.randn(2, 3, 224, 224).cuda()
        gt = torch.rand_like(x, requires_grad=True)
        y = model(x)
        loss = (y - gt).mean().backward()

        # STEP 3
        # IMPLEMENTATION #1
        loss_sim_tracker = LossSimTracker(model)
        loss_sim_1 = loss_sim_tracker.compute_loss_sim()

        loss_sim_2 = compute_loss_sim(model)

        print(loss_sim_1.item(), loss_sim_2.item())
