import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=64, alpha=16, dropout=0.05):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r
        
    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scaling