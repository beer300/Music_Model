import torch
from torch import nn

class ResidualLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x += residual
        return self.relu(x)

class VQVAEEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256, 256, 256, 256]):
        super().__init__()
        layers = []
        prev_channels = in_channels
        self.conv1d_1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv1d_3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv1d_4 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv1d_5 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        # Build six convolutional layers
        
            
        self.conv_layers = nn.Sequential(*layers)
        self.residual = ResidualLayer()

    def forward(self, x):
        x_conv_1 = self.conv1d_1(x)
        x_conv_2 = self.conv1d_1(x_conv_1) + x_conv_1
        x_conv_3 = self.conv1d_1(x_conv_2)
        x_conv_4 = self.conv1d_1(x_conv_3) + x_conv_3
        x_conv_5 = self.conv1d_1(x_conv_4) + x_conv_4
        output = self.residual(x_conv_5)
        return output 

# Example usage
if __name__ == "__main__":
    encoder = VQVAEEncoder()
    input_tensor = torch.randn(1, 3, 64, 64)  # (batch, channels, height, width)
    output = encoder(input_tensor)
    print(f"Output shape: {output.shape}")  # Should be (1, 256, 1, 1) for 64x64 input