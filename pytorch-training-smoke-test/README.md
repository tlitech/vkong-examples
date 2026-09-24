# Train a small CNN with PyTorch

Train a small CNN on MNIST for two epochs and print its test accuracy. Torchvision downloads the dataset automatically. This is a short check of GPU training and task completion.

## Run

From this repository's root, after `vkong login`:

```bash
cd pytorch-training-smoke-test
vkong run
```

The task prints its result and exits. Its `mnist_cnn.pt` stays in the remote
project directory while the machine is active.

## Keep the output

For durable output, add `storage: pytorch-training-smoke-test-data` to `vkong.yaml` and
change the output path in the Python script to `/data/mnist_cnn.pt` before running.
Allow enough `disk_gb` for the runtime and restored data. After a successful run,
stop the App to save the volume, then download the saved file:

```bash
vkong app stop pytorch-training-smoke-test
vkong storage download pytorch-training-smoke-test-data mnist_cnn.pt -o mnist_cnn.pt
```

Storage only saves files under `/data`. The default project-directory output
is temporary and is removed when the machine is destroyed.

## Stop

```bash
vkong app stop pytorch-training-smoke-test
```

For a smoke run whose output can be discarded, use `vkong run --auto-stop`
to release compute when the task finishes.
