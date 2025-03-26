# Advanced-DGM-PyTorch

Implementation of Deep Generative Models in PyTorch (VQ-VAE/Diffusion Model/Flow Model)

## Requisite

- Python >= 3.10
- PyTorch >= 2.0

## How to run:
- choose `MODEL NAME` in [`vqvae`]
- choose `DATA NAME` in ['mnist', 'cifar']
- run the command:

```bash
python main.py --dgm {MODEL NAME} --data {DATA NAME}
```

### Reconstruction Samples (VQ-VAE)

- MNIST

![mnist_original](https://github.com/sungbinlim/Advanced-DGM-PyTorch/blob/main/imgs/MNIST_original_imgs.png)

![mnist_recon](https://github.com/sungbinlim/Advanced-DGM-PyTorch/blob/main/imgs/MNIST_recon_imgs.png)

- CelebA


![celeba_original](https://github.com/sungbinlim/Advanced-DGM-PyTorch/blob/main/imgs/original_generated_images.png)

![celeba_recon](https://github.com/sungbinlim/Advanced-DGM-PyTorch/blob/main/imgs/reconstructed_generated_images.png)


## License

```
MIT License

Copyright (c) 2025 Korea University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
