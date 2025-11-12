import torch.nn

class Neural_Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.network_stack = torch.nn.Sequential(
            torch.nn.Linear(in_features = 28 * 28, out_features = 512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = 512, out_features = 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.network_stack(x)
        return output