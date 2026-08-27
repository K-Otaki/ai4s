#!/usr/bin/env python3

import os
import time
import random

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

SEED = 42

GLOBAL_BATCH_SIZE = 128

EPOCHS = 5
LEARNING_RATE = 1.0e-3

NUM_WORKERS = 2

INFERENCE_WARMUP_BATCHES = 5
INFERENCE_REPEATS = 5


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # More reproducible behavior.
    # This may slightly reduce performance.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Neural network
# ============================================================

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            1,
            32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(
            64 * 7 * 7,
            128
        )

        self.fc2 = nn.Linear(
            128,
            10
        )

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(
            self.conv1(x)
        )

        x = self.pool(x)

        x = self.relu(
            self.conv2(x)
        )

        x = self.pool(x)

        x = torch.flatten(
            x,
            1
        )

        x = self.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x


# ============================================================
# Distributed setup
# ============================================================

def setup_distributed():

    distributed = "RANK" in os.environ

    if distributed:

        rank = int(
            os.environ["RANK"]
        )

        local_rank = int(
            os.environ["LOCAL_RANK"]
        )

        world_size = int(
            os.environ["WORLD_SIZE"]
        )

        torch.cuda.set_device(
            local_rank
        )

        device = torch.device(
            f"cuda:{local_rank}"
        )

        dist.init_process_group(
            backend="nccl",
            device_id=device
        )

    else:

        rank = 0
        local_rank = 0
        world_size = 1

        torch.cuda.set_device(0)

        device = torch.device(
            "cuda:0"
        )

    return (
        rank,
        local_rank,
        world_size,
        device,
        distributed
    )


# ============================================================
# Synchronization helper
# ============================================================

def synchronize(distributed):

    if distributed:
        dist.barrier()

    torch.cuda.synchronize()


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Distributed initialization
    # --------------------------------------------------------

    (
        rank,
        local_rank,
        world_size,
        device,
        distributed
    ) = setup_distributed()

    # --------------------------------------------------------
    # Random seed
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Per-GPU batch size
    # --------------------------------------------------------

    if GLOBAL_BATCH_SIZE % world_size != 0:

        if rank == 0:
            print(
                "ERROR: GLOBAL_BATCH_SIZE must be "
                "divisible by WORLD_SIZE."
            )

        if distributed:
            dist.destroy_process_group()

        return

    local_batch_size = (
        GLOBAL_BATCH_SIZE
        // world_size
    )

    # --------------------------------------------------------
    # System information
    # --------------------------------------------------------

    if rank == 0:

        print("=" * 70)
        print("MNIST GPU training")
        print("=" * 70)

        print(
            f"Execution mode     : "
            f"{'DDP' if distributed else 'Single GPU'}"
        )

        print(
            f"World size         : "
            f"{world_size}"
        )

        print(
            f"Global batch size  : "
            f"{GLOBAL_BATCH_SIZE}"
        )

        print(
            f"Batch size / GPU   : "
            f"{local_batch_size}"
        )

        print(
            f"Random seed        : "
            f"{SEED}"
        )

        print(
            f"PyTorch            : "
            f"{torch.__version__}"
        )

        print(
            f"CUDA               : "
            f"{torch.version.cuda}"
        )

        print()

    print(
        f"Rank {rank}: "
        f"host={os.uname().nodename}, "
        f"local_rank={local_rank}, "
        f"GPU={torch.cuda.get_device_name(local_rank)}",
        flush=True
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    transform = transforms.Compose(
        [
            transforms.ToTensor(),

            transforms.Normalize(
                (0.1307,),
                (0.3081,)
            )
        ]
    )

    data_dir = "./data"

    # Only rank 0 downloads
    if rank == 0:

        datasets.MNIST(
            data_dir,
            train=True,
            download=True,
            transform=transform
        )

        datasets.MNIST(
            data_dir,
            train=False,
            download=True,
            transform=transform
        )

    if distributed:
        dist.barrier()

    train_dataset = datasets.MNIST(
        data_dir,
        train=True,
        download=False,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        data_dir,
        train=False,
        download=False,
        transform=transform
    )

    # --------------------------------------------------------
    # Distributed sampler
    # --------------------------------------------------------

    if distributed:

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=SEED
        )

    else:

        train_sampler = None

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,

        batch_size=local_batch_size,

        sampler=train_sampler,

        shuffle=(
            train_sampler is None
        ),

        num_workers=NUM_WORKERS,

        pin_memory=True,

        generator=generator
    )

    test_loader = DataLoader(
        test_dataset,

        batch_size=256,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=True
    )

    if rank == 0:

        print(
            f"Training samples   : "
            f"{len(train_dataset)}"
        )

        print(
            f"Test samples       : "
            f"{len(test_dataset)}"
        )

        print()

    local_samples = (
        len(train_sampler)
        if train_sampler is not None
        else len(train_dataset)
    )

    print(
        f"Rank {rank}: "
        f"training samples assigned = "
        f"{local_samples}",
        flush=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = Net().to(
        device
    )

    if distributed:

        model = DDP(
            model,
            device_ids=[
                local_rank
            ]
        )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    train_losses = []
    train_accuracies = []
    epoch_times = []

    # ========================================================
    # Training
    # ========================================================

    synchronize(
        distributed
    )

    training_start = (
        time.perf_counter()
    )

    for epoch in range(EPOCHS):

        if distributed:

            train_sampler.set_epoch(
                epoch
            )

        model.train()

        local_loss = 0.0
        local_correct = 0
        local_total = 0

        synchronize(
            distributed
        )

        epoch_start = (
            time.perf_counter()
        )

        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                images
            )

            loss = criterion(
                output,
                labels
            )

            loss.backward()

            optimizer.step()

            batch_size = (
                labels.size(0)
            )

            local_loss += (
                loss.item()
                * batch_size
            )

            predicted = (
                output.argmax(
                    dim=1
                )
            )

            local_correct += (
                predicted
                == labels
            ).sum().item()

            local_total += (
                batch_size
            )

        # ----------------------------------------------------
        # Global statistics
        # ----------------------------------------------------

        statistics = torch.tensor(
            [
                local_loss,
                local_correct,
                local_total
            ],

            dtype=torch.float64,

            device=device
        )

        if distributed:

            dist.all_reduce(
                statistics,
                op=dist.ReduceOp.SUM
            )

        global_loss = (
            statistics[0].item()
            / statistics[2].item()
        )

        global_accuracy = (
            100.0
            * statistics[1].item()
            / statistics[2].item()
        )

        synchronize(
            distributed
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        if rank == 0:

            train_losses.append(
                global_loss
            )

            train_accuracies.append(
                global_accuracy
            )

            epoch_times.append(
                epoch_time
            )

            print(
                f"Epoch {epoch + 1:2d}/{EPOCHS}: "
                f"loss={global_loss:.4f}, "
                f"accuracy={global_accuracy:.2f}%, "
                f"time={epoch_time:.3f} s"
            )

    # --------------------------------------------------------
    # Total training time
    # --------------------------------------------------------

    synchronize(
        distributed
    )

    training_time = (
        time.perf_counter()
        - training_start
    )

    # ========================================================
    # Inference
    #
    # Only rank 0 performs inference.
    # Therefore inference always uses one GPU.
    # ========================================================

    if rank == 0:

        print()
        print("=" * 70)
        print("Training completed")
        print("=" * 70)

        print(
            f"Total training time: "
            f"{training_time:.3f} s"
        )

        if distributed:
            inference_model = (
                model.module
            )
        else:
            inference_model = model

        inference_model.eval()

        # ====================================================
        # Warm-up
        # ====================================================

        with torch.no_grad():

            for batch_index, (
                images,
                labels
            ) in enumerate(
                test_loader
            ):

                images = images.to(
                    device,
                    non_blocking=True
                )

                _ = inference_model(
                    images
                )

                if (
                    batch_index + 1
                    >= INFERENCE_WARMUP_BATCHES
                ):
                    break

        torch.cuda.synchronize()

        # ====================================================
        # Repeated inference benchmark
        # ====================================================

        inference_times = []

        final_correct = 0
        final_total = 0

        sample_images = None
        sample_labels = None
        sample_predictions = None

        for repeat in range(
            INFERENCE_REPEATS
        ):

            correct = 0
            total = 0

            torch.cuda.synchronize()

            inference_start = (
                time.perf_counter()
            )

            with torch.no_grad():

                for batch_index, (
                    images,
                    labels
                ) in enumerate(
                    test_loader
                ):

                    images = images.to(
                        device,
                        non_blocking=True
                    )

                    labels = labels.to(
                        device,
                        non_blocking=True
                    )

                    output = (
                        inference_model(
                            images
                        )
                    )

                    prediction = (
                        output.argmax(
                            dim=1
                        )
                    )

                    correct += (
                        prediction
                        == labels
                    ).sum().item()

                    total += (
                        labels.size(0)
                    )

                    # Store first 16 images
                    # only on first repeat
                    if (
                        repeat == 0
                        and batch_index == 0
                    ):

                        sample_images = (
                            images[:16]
                            .cpu()
                        )

                        sample_labels = (
                            labels[:16]
                            .cpu()
                        )

                        sample_predictions = (
                            prediction[:16]
                            .cpu()
                        )

            torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                - inference_start
            )

            inference_times.append(
                elapsed
            )

            if repeat == 0:

                final_correct = correct
                final_total = total

    # --------------------------------------------------------
    # Important:
    # rank 1 must wait here until rank 0 finishes inference.
    # --------------------------------------------------------

    if distributed:
        dist.barrier()

    # ========================================================
    # Rank 0 result output
    # ========================================================

    if rank == 0:

        inference_times = np.array(
            inference_times
        )

        inference_mean = (
            inference_times.mean()
        )

        inference_std = (
            inference_times.std()
        )

        accuracy = (
            100.0
            * final_correct
            / final_total
        )

        print()

        print(
            f"Test accuracy      : "
            f"{accuracy:.2f}%"
        )

        print(
            f"Inference repeats  : "
            f"{INFERENCE_REPEATS}"
        )

        print(
            f"Inference time mean: "
            f"{inference_mean:.4f} s"
        )

        print(
            f"Inference time std : "
            f"{inference_std:.4f} s"
        )

        print(
            f"Images             : "
            f"{final_total}"
        )

        print(
            f"Time / image       : "
            f"{1.0e6 * inference_mean / final_total:.2f} us"
        )

        print(
            f"Throughput         : "
            f"{final_total / inference_mean:.1f} images/s"
        )

        # ====================================================
        # Training curve
        # ====================================================

        epochs_array = np.arange(
            1,
            EPOCHS + 1
        )

        fig, ax1 = plt.subplots(
            figsize=(7, 5)
        )

        ax1.plot(
            epochs_array,
            train_losses,
            marker="o",
            label="Loss"
        )

        ax1.set_xlabel(
            "Epoch"
        )

        ax1.set_ylabel(
            "Training loss"
        )

        ax1.grid(
            True
        )

        # ax2 = ax1.twinx()

        # ax2.plot(
        #     epochs_array,
        #     train_accuracies,
        #     marker="s",
        #     label="Accuracy"
        # )

        # ax2.set_ylabel(
        #     "Training accuracy (%)"
        # )

        plt.title(
            "MNIST training curve"
        )

        fig.tight_layout()

        plt.savefig(
            "fig/mnist_training_curve.png",
            dpi=150
        )

        plt.close(
            fig
        )

        # ====================================================
        # Epoch time plot
        # ====================================================

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        ax.plot(
            epochs_array,
            epoch_times,
            marker="o"
        )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "Time [s]"
        )

        ax.set_title(
            "Training time per epoch"
        )

        ax.grid(
            True
        )

        fig.tight_layout()

        plt.savefig(
            "fig/mnist_epoch_time.png",
            dpi=150
        )

        plt.close(
            fig
        )

        # ====================================================
        # Prediction visualization
        # ====================================================

        fig, axes = plt.subplots(
            4,
            4,
            figsize=(8, 8)
        )

        for i, ax in enumerate(
            axes.flat
        ):

            image = (
                sample_images[i]
                .squeeze()
                * 0.3081
                + 0.1307
            )

            ax.imshow(
                image,
                cmap="gray"
            )

            true_label = (
                sample_labels[i]
                .item()
            )

            predicted_label = (
                sample_predictions[i]
                .item()
            )

            if (
                true_label
                == predicted_label
            ):

                title = (
                    f"True: {true_label}\n"
                    f"Pred: {predicted_label}"
                )

            else:

                title = (
                    f"True: {true_label}\n"
                    f"Pred: {predicted_label} (wrong)"
                )

            ax.set_title(
                title
            )

            ax.axis(
                "off"
            )

        plt.tight_layout()

        plt.savefig(
            "fig/mnist_predictions.png",
            dpi=150
        )

        plt.close(
            fig
        )

        # ====================================================
        # Save model
        # ====================================================

        torch.save(
            inference_model.state_dict(),
            "models/mnist.pt"
        )

        print()
        print("Output files:")
        print("  fig/training_curve.png")
        print("  fig/epoch_time.png")
        print("  fig/predictions.png")
        print("  models/mnist.pt")

    # ========================================================
    # Cleanup
    # ========================================================

    if distributed:

        dist.destroy_process_group()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()