# Train MNIST

This finite task trains a small CNN for two epochs on MNIST, prints test accuracy,
saves `mnist_cnn.pt`, and exits with the Python process status.

```bash
cd vkong-examples/mnist_training
vkong run -C .
```

Use `--detach` to leave the training process running after the CLI returns. Add
`--auto-stop` when vkong should destroy the GPU as soon as training finishes:

```bash
vkong run -C . --detach --auto-stop
```

The current CLI retains the training logs but does not download checkpoints yet. Without
`--auto-stop`, `mnist_cnn.pt` remains in the remote workspace while the rental is active.
