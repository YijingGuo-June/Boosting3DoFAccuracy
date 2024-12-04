import re

import torch


class ScaledLowRankConvAdapter(torch.nn.Module):
    """SLR adapter for conv layers."""

    def __init__(self, conv2d: torch.nn.Conv2d, hidden_dim: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = conv2d
        self.kernel_size = conv2d.kernel_size
        self.scaler = torch.nn.Parameter(torch.ones(self.proj.out_channels))

        assert conv2d.kernel_size == (14, 14)
        kernel_size = (2, 2)
        self.down = torch.nn.Conv2d(
            self.proj.in_channels,
            self.hidden_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
        )
        self.up = torch.nn.Conv2d(
            self.hidden_dim,
            self.proj.out_channels,
            # kernel_size=kernel_size,
            # stride=kernel_size,
            kernel_size=(7, 7),
            stride=(7, 7),
        )

        for p in self.proj.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print('x.shape:', x.shape)
        x_lr = self.up(self.down(x))
        x = self.proj(x)

        x += x_lr

        return torch.einsum("bdhw,d->bdhw", x, self.scaler)


class ScaledLowRankAdapter(torch.nn.Module):
    """SLR adapter for linear layers.

    Adds a low rank adapter and scaling parameters to a linear layer"
    """

    def __init__(self, linear: torch.nn.Linear, hidden_dim: int = 16):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.linear = linear
        self.out_dim, self.in_dim = self.linear.weight.shape

        # freeze original parameters
        for p in self.linear.parameters():
            p.requires_grad = False

        # initialize scaling vectors as ones
        self.in_scaler = torch.nn.Parameter(torch.ones(self.in_dim))
        self.out_scaler = torch.nn.Parameter(torch.ones(self.out_dim))

        self.down = torch.nn.Linear(self.in_dim, self.hidden_dim)
        self.up = torch.nn.Linear(self.hidden_dim, self.out_dim)

        # init low-rank matrices as normal/zeros
        self.up.weight.data.fill_(0)
        self.up.bias.data.fill_(0)
        torch.nn.init.normal_(self.down.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x *= self.in_scaler
        # x_lr = self.up(self.down(x))
        # x = self.linear(x)
        # x += x_lr
        # x *= self.out_scaler

        # without in-place operations (chaned due to torch error, version above used for most experiments)
        x_scaled = x * self.in_scaler
        x_lr = self.up(self.down(x_scaled))
        x = self.linear(x_scaled)
        x_new = x + x_lr
        x = x_new * self.out_scaler

        # return x + x_lr
        return x

class ScaledLowRankConfigTimmViT:
    """Config to add SLR adapters to a timm ViT."""

    def __init__(
        self, hidden_dim: int = 8, patch_embed: bool = False, norm: bool = True
    ):
        self.lora_rank = hidden_dim
        self.adapter_modules = ".*attn|.*mlp|decoder_embed|decoder_pred"
        # # nn.leaner named fc in DINOv2, lin in SAM
        self.adapter_layers = "qkv|fc1|fc2|lin1|lin2|proj|decoder_embed|decoder_pred"
        if patch_embed:
            self.adapter_modules += "|patch_embed"
            self.adapter_layers += "|proj"
        self.model_modifier = "adapter"
        self.extra_trainable_param_names = "fcn_high.pred.7|fcn_low.pred.7"
        if norm:
            self.extra_trainable_param_names += "|.*norm.*"
        # self.extra_trainable_param_names = ".*norm.*|.*decoder_embed.*|.*decoder_pred.*"
        # self.extra_trainable_param_names = ".*norm.*|.*decoder_embed.*"

"""
usage:
model = add_extra_weights(
    model,
    config,
    ScaledLowRankAdapter,
    ScaledLowRankConvAdapter,
    adapter_trainable,
    only_scaler_trainable,
)
"""
def add_extra_weights(
    model,
    config,
    adapter,
    conv_adapter=None,
    trainable=True,
    only_scaler_trainable=False,
):
    # together with config of type ScaledLowRankConfigTimmViT
    for m_name, module in dict(model.named_modules()).items():
        if re.fullmatch(config.adapter_modules, m_name):
            children = dict(module.named_children())
            set_as_module = False
            if not children:
                set_as_module = True
                # if module is a layer
                children = {m_name: module}
            for c_name, layer in children.items():
                if re.fullmatch(config.adapter_layers, c_name):
                    if isinstance(layer, torch.nn.Linear):
                        adp = adapter
                    elif isinstance(layer, torch.nn.Conv2d):
                        if conv_adapter is None:
                            continue
                        adp = conv_adapter
                    else:
                        raise ValueError()
                    adapter_instance = adp(layer, hidden_dim=config.lora_rank)
                    if not trainable:
                        for p in adapter_instance.parameters():
                            p.requires_grad = False
                    if only_scaler_trainable:
                        for n, p in adapter_instance.named_parameters():
                            if "scaler" in n:
                                p.requires_grad = True
                            else:
                                p.requires_grad = False
                    if set_as_module:
                        setattr(model, c_name, adapter_instance)
                    else:
                        setattr(module, c_name, adapter_instance)

        # make extra params trainable (e.g., layer norm layers)
        if re.fullmatch(config.extra_trainable_param_names, m_name):
            for p in module.parameters():
                p.requires_grad = True

    return model
