# GPU training smoke test with PyTorch

Use this finite task to verify VKong's complete GPU training path: provision a GPU,
run a small PyTorch CNN for two epochs, save a checkpoint, and exit cleanly. MNIST
keeps the test fast and inexpensive; it is the test workload, not the point of the recipe.

```bash
cd vkong-examples/pytorch-training-smoke-test
vkong run -C .
```

Use `--detach` to leave the training process running after the CLI returns. Add
`--auto-stop` when vkong should destroy the GPU as soon as training finishes:

```bash
vkong run -C . --detach --auto-stop
```

The current CLI retains the training logs but does not download checkpoints yet. Without
`--auto-stop`, `mnist_cnn.pt` remains in the remote workspace while the rental is active.
