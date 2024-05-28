# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Loading data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, 
    batch_size=64, shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, 
    batch_size=64, 
    shuffle=False,
)


class LinearAutoencoder(nn.Module):
    """_summary_: In this implementation we will use six hidden layers.
    The decoder is essentially the inverse of the encoder. 
    """
    def __init__(self):
        super(LinearAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 16), 
            nn.Linear(16, 3) 
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.Linear(16, 64),
            nn.Linear(64, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid()  # Use sigmoid activation to output values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



def train(num_epochs, criterion, optimizer, model):

  for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1)
        output = model(img)
        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
def reconstruct():
    
  with torch.no_grad():
    for data in test_loader:
        img, _ = data
        img = img.view(img.size(0), -1)
        output = model(img)

        # Plotting the first 8 test images and their reconstructions
        fig, axes = plt.subplots( 2, 8, figsize=(12, 3))
        for i in range(8):
            axes[0, i].imshow(img[i].view(28, 28).numpy(), cmap='gray')
            axes[1, i].imshow(output[i].view(28, 28).numpy(), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].axis('off')
        plt.show()
        break


if __name__ == "__main__":
    
    model = LinearAutoencoder()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    
    train(num_epochs, criterion, optimizer, model)
    reconstruct()
    
    

