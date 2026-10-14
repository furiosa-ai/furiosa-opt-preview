# MNIST example

A minimal fully-connected MNIST classifier (784 → 256 ReLU → 10):

- `mod.rs` — VISA kernel (`forward`).
- `../../data/mnist/mnist.safetensors` — trained bf16 weights and ten MNIST test images with their labels.
- `../../tests/mnist_tests.rs` — runs `forward` on each image and checks the predicted digit against its label.

The kernel works on padded extents: the input is 784 padded to 800 (`X`), and the output is 10 padded to 16 (`C`).
The `hw.*` tensors in `mnist.safetensors` hold the padded weights and images.

## Run

```bash
cargo furiosa-opt test --test mnist_tests -- --nocapture
```
