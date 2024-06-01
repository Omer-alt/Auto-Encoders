# Implementation in pytorch of Auto-Encoders: Linear, Denoising, Contractive Auto-encoders

### 1. Data Loading
MNIST dataset was loading with batch size of 64 and intermediates operations like Normalisation () 

## I. Linear Auto-Encoders
In this case of Deep linear autoencoder, when we use six hidden layers with the following linear structure inside the encoder: 28 * 28 -> 512 -> 256 -> 128 -> 64 -> 16 -> 3, the performance of the model is very poor because the size of the latent code is reduced, Network architecture is too deep Compare to the Simple Linear Autoencoder model which performs better despite its simple architecture: 28 * 28 -> 128 -> 64 -> 32 -> 16 -> 8 

**Result:** 

![Linear_auto_encoder](./assets/Linear_auto_encoder.png)
Deep Linear auto-encoder reconstruction of eigth MNIST images

![Linear_auto_encoder](./assets/simple_linear.png)
Simple Linear auto-encoder reconstruction of eigth MNIST images

## II. Denoising Auto-Encoders
## III. Contractive Auto-Encoders
For the contractive autoencoder, we use as architecture of the encoder two fully connected layers with relu as activation function and 


**Result:** 

![Contractive_auto_encoder](./assets/Img_lambda=5.png)
Contractive auto-encoder reconstruction of eigth MNIST images for lambda = 5

![Contractive_auto_encoder](./assets/Img_lambda_very_small.png)
Contractive auto-encoder reconstruction of eigth MNIST images for lambda = 1e-4

### Build With

**Language:** Python

**Package:** torchvision, matplotlib, Pytorch

### Run Locally
### License

[MIT](https://choosealicense.com/licenses/mit/)