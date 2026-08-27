import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

nx = 64
ny = 64

kappa = 0.01

dx = 1.0 / (nx - 1)
dy = 1.0 / (ny - 1)

# Explicit finite-difference stability condition:
#
# kappa * dt * (1/dx^2 + 1/dy^2) <= 1/2
#
dt = 0.2 / (kappa * (1.0 / dx**2 + 1.0 / dy**2))

n_steps = 100

n_train = 4000
n_test = 400

global_batch_size = 32
n_epochs = 100

learning_rate = 1e-3

# FNO parameters
modes1 = 12
modes2 = 12
width = 32

seed = 42


# ============================================================
# Distributed setup
# ============================================================

def setup_distributed():

    distributed = "RANK" in os.environ

    if distributed:

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Distributed execution requires CUDA GPUs."
            )

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        device = torch.device(
            f"cuda:{local_rank}"
        )

        dist.init_process_group(
            backend="nccl",
            device_id=device,
        )

    else:

        rank = 0
        local_rank = 0
        world_size = 1

        device = torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    return (
        rank,
        local_rank,
        world_size,
        device,
        distributed,
    )


(
    rank,
    local_rank,
    world_size,
    device,
    distributed,
) = setup_distributed()


if global_batch_size % world_size != 0:
    raise ValueError(
        "global_batch_size must be divisible by WORLD_SIZE."
    )

local_batch_size = global_batch_size // world_size

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

if rank == 0:
    print("PyTorch        :", torch.__version__)
    print(
        "Execution mode :",
        "DDP" if distributed else "Single process",
    )
    print("World size     :", world_size)
    print("Global batch   :", global_batch_size)
    print("Batch per GPU  :", local_batch_size)

print(
    f"Rank {rank}: host={os.uname().nodename}, "
    f"local_rank={local_rank}, device={device}",
    flush=True,
)

if device.type == "cuda":
    print(
        f"Rank {rank}: GPU="
        f"{torch.cuda.get_device_name(local_rank)}",
        flush=True,
    )


# ============================================================
# Grid
# ============================================================

x = np.linspace(0.0, 1.0, nx)
y = np.linspace(0.0, 1.0, ny)

X, Y = np.meshgrid(
    x,
    y,
    indexing="ij",
)


# ============================================================
# Random initial condition
# ============================================================

def random_initial_condition():
    """
    Random smooth initial field satisfying
    u = 0 on all boundaries.

    u(x,y) = sum a_nm sin(n*pi*x) sin(m*pi*y)
    """

    u = np.zeros((nx, ny), dtype=np.float32)

    n_modes = 5

    for n in range(1, n_modes + 1):
        for m in range(1, n_modes + 1):

            amplitude = np.random.uniform(
                -1.0,
                1.0,
            )

            u += (
                amplitude
                * np.sin(n * np.pi * X)
                * np.sin(m * np.pi * Y)
            )

    # Dirichlet boundary condition
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0

    return u.astype(np.float32)


# ============================================================
# 2D heat equation solver
# ============================================================

def evolve_heat(u0):
    """
    Solve

        du/dt = kappa * (d2u/dx2 + d2u/dy2)

    with explicit finite differences.
    """

    u = u0.copy()

    rx = kappa * dt / dx**2
    ry = kappa * dt / dy**2

    for _ in range(n_steps):

        unew = u.copy()

        unew[1:-1, 1:-1] = (
            u[1:-1, 1:-1]

            + rx
            * (
                u[2:, 1:-1]
                - 2.0 * u[1:-1, 1:-1]
                + u[:-2, 1:-1]
            )

            + ry
            * (
                u[1:-1, 2:]
                - 2.0 * u[1:-1, 1:-1]
                + u[1:-1, :-2]
            )
        )

        # Dirichlet boundary conditions
        unew[0, :] = 0.0
        unew[-1, :] = 0.0
        unew[:, 0] = 0.0
        unew[:, -1] = 0.0

        u = unew

    return u.astype(np.float32)


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(n_samples):

    inputs = np.empty(
        (n_samples, nx, ny),
        dtype=np.float32,
    )

    targets = np.empty(
        (n_samples, nx, ny),
        dtype=np.float32,
    )

    for i in range(n_samples):

        u0 = random_initial_condition()
        uT = evolve_heat(u0)

        inputs[i] = u0
        targets[i] = uT

        if rank == 0 and (i + 1) % 500 == 0:
            print(
                f"Generated {i + 1}/{n_samples}"
            )

    return inputs, targets


if rank == 0:
    print("Generating training data...")

X_train, Y_train = generate_dataset(
    n_train
)

if rank == 0:
    print("Generating test data...")

X_test, Y_test = generate_dataset(
    n_test
)


# ============================================================
# Normalization
# ============================================================

mean = X_train.mean()
std = X_train.std()

X_train_normalized = (
    X_train - mean
) / std

X_test_normalized = (
    X_test - mean
) / std

Y_train_normalized = (
    Y_train - mean
) / std

Y_test_normalized = (
    Y_test - mean
) / std


# ============================================================
# PyTorch datasets
# ============================================================

train_dataset = TensorDataset(
    torch.from_numpy(
        X_train_normalized
    ),
    torch.from_numpy(
        Y_train_normalized
    ),
)

test_dataset = TensorDataset(
    torch.from_numpy(
        X_test_normalized
    ),
    torch.from_numpy(
        Y_test_normalized
    ),
)

if distributed:
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
    )

    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
else:
    train_sampler = None
    test_sampler = None

train_loader = DataLoader(
    train_dataset,
    batch_size=local_batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
    pin_memory=(device.type == "cuda"),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=local_batch_size,
    sampler=test_sampler,
    shuffle=False,
    pin_memory=(device.type == "cuda"),
)


# ============================================================
# Spectral convolution
# ============================================================

class SpectralConv2d(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        modes1,
        modes2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes1 = modes1
        self.modes2 = modes2

        if modes1 > nx // 2:
            raise ValueError(
                "modes1 must be at most nx // 2."
            )

        if modes2 > ny // 2 + 1:
            raise ValueError(
                "modes2 must be at most ny // 2 + 1."
            )

        scale = 1.0 / (
            in_channels * out_channels
        )

        self.weights1 = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes1,
                modes2,
                dtype=torch.cfloat,
            )
        )

        self.weights2 = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes1,
                modes2,
                dtype=torch.cfloat,
            )
        )

        # The CUDA 13.3 container used on the target GH200
        # system fails in cuFFT.  Precompute only the Fourier
        # modes used by the FNO and evaluate the transforms as
        # separable matrix products instead of torch.fft calls.
        kx = torch.cat(
            (
                torch.arange(
                    modes1,
                    dtype=torch.float32,
                ),
                torch.arange(
                    -modes1,
                    0,
                    dtype=torch.float32,
                ),
            )
        )
        ky = torch.arange(
            modes2,
            dtype=torch.float32,
        )

        grid_x = torch.arange(
            nx,
            dtype=torch.float32,
        )
        grid_y = torch.arange(
            ny,
            dtype=torch.float32,
        )

        forward_x = torch.exp(
            -2j
            * torch.pi
            * kx[:, None]
            * grid_x[None, :]
            / nx
        )
        forward_y = torch.exp(
            -2j
            * torch.pi
            * ky[:, None]
            * grid_y[None, :]
            / ny
        )

        inverse_factors = torch.full(
            (modes2,),
            2.0,
            dtype=torch.float32,
        )
        inverse_factors[0] = 1.0

        if ny % 2 == 0 and modes2 == ny // 2 + 1:
            inverse_factors[-1] = 1.0

        self.register_buffer(
            "forward_x",
            forward_x,
            persistent=False,
        )
        self.register_buffer(
            "forward_y",
            forward_y,
            persistent=False,
        )
        self.register_buffer(
            "inverse_x",
            forward_x.conj(),
            persistent=False,
        )
        self.register_buffer(
            "inverse_y",
            forward_y.conj(),
            persistent=False,
        )
        self.register_buffer(
            "inverse_factors",
            inverse_factors,
            persistent=False,
        )


    def complex_mul(
        self,
        input,
        weights,
    ):

        return torch.einsum(
            "bixy,ioxy->boxy",
            input,
            weights,
        )


    def forward(self, x):

        # Forward transform, restricted to the retained modes.
        transformed_y = torch.einsum(
            "bcxy,ky->bcxk",
            x.to(torch.cfloat),
            self.forward_y,
        )
        retained_modes = torch.einsum(
            "bcxk,jx->bcjk",
            transformed_y,
            self.forward_x,
        )

        positive_modes = self.complex_mul(
            retained_modes[
                :,
                :,
                : self.modes1,
                :,
            ],
            self.weights1,
        )
        negative_modes = self.complex_mul(
            retained_modes[
                :,
                :,
                self.modes1 :,
                :,
            ],
            self.weights2,
        )

        output_modes = torch.cat(
            (
                positive_modes,
                negative_modes,
            ),
            dim=2,
        )

        # Inverse transform. Positive y frequencies except DC
        # (and Nyquist, when retained) represent conjugate pairs.
        output_modes = (
            output_modes
            * self.inverse_factors[
                None,
                None,
                None,
                :,
            ]
        )
        transformed_x = torch.einsum(
            "bojk,jx->boxk",
            output_modes,
            self.inverse_x,
        )
        output = torch.einsum(
            "boxk,ky->boxy",
            transformed_x,
            self.inverse_y,
        )

        return output.real / (nx * ny)


# ============================================================
# FNO layer
# ============================================================

class FNOBlock(nn.Module):

    def __init__(
        self,
        width,
        modes1,
        modes2,
    ):
        super().__init__()

        self.spectral = SpectralConv2d(
            width,
            width,
            modes1,
            modes2,
        )

        self.local = nn.Conv2d(
            width,
            width,
            kernel_size=1,
        )

        self.activation = nn.GELU()


    def forward(self, x):

        x1 = self.spectral(x)
        x2 = self.local(x)

        return self.activation(
            x1 + x2
        )


# ============================================================
# FNO model
# ============================================================

class FNO2d(nn.Module):

    def __init__(
        self,
        modes1,
        modes2,
        width,
    ):
        super().__init__()

        # Input channels:
        #
        # 0 : u(x,y)
        # 1 : x coordinate
        # 2 : y coordinate

        self.input_projection = nn.Conv2d(
            3,
            width,
            kernel_size=1,
        )

        self.fno1 = FNOBlock(
            width,
            modes1,
            modes2,
        )

        self.fno2 = FNOBlock(
            width,
            modes1,
            modes2,
        )

        self.fno3 = FNOBlock(
            width,
            modes1,
            modes2,
        )

        self.fno4 = FNOBlock(
            width,
            modes1,
            modes2,
        )

        self.output_projection1 = nn.Conv2d(
            width,
            128,
            kernel_size=1,
        )

        self.output_projection2 = nn.Conv2d(
            128,
            1,
            kernel_size=1,
        )

        self.activation = nn.GELU()


    def forward(self, u):

        batch_size = u.shape[0]

        # Coordinate grid
        grid_x = torch.linspace(
            0.0,
            1.0,
            nx,
            device=u.device,
        )

        grid_y = torch.linspace(
            0.0,
            1.0,
            ny,
            device=u.device,
        )

        gx, gy = torch.meshgrid(
            grid_x,
            grid_y,
            indexing="ij",
        )

        gx = gx.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

        gy = gy.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

        # [B, 3, nx, ny]
        x = torch.stack(
            (
                u,
                gx,
                gy,
            ),
            dim=1,
        )

        x = self.input_projection(x)

        x = self.fno1(x)
        x = self.fno2(x)
        x = self.fno3(x)
        x = self.fno4(x)

        x = self.activation(
            self.output_projection1(x)
        )

        x = self.output_projection2(x)

        return x[:, 0]


# ============================================================
# Model
# ============================================================

model = FNO2d(
    modes1=modes1,
    modes2=modes2,
    width=width,
).to(device)

if distributed:
    model = DDP(
        model,
        device_ids=[local_rank],
    )

if rank == 0:
    print(model)

n_parameters = sum(
    p.numel()
    for p in model.parameters()
)

if rank == 0:
    print(
        f"Number of parameters: "
        f"{n_parameters:,}"
    )
    print(
        "Spectral transform: direct low-mode DFT "
        "(cuFFT disabled)"
    )


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.5,
)

criterion = nn.MSELoss()


# ============================================================
# Training
# ============================================================

for epoch in range(n_epochs):

    if distributed:
        train_sampler.set_epoch(epoch)

    model.train()

    local_train_loss = 0.0
    local_train_count = 0

    for input_batch, target_batch in train_loader:

        input_batch = input_batch.to(
            device,
            non_blocking=True,
        )
        target_batch = target_batch.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        prediction = model(
            input_batch
        )

        loss = criterion(
            prediction,
            target_batch,
        )

        loss.backward()

        optimizer.step()

        local_train_loss += (
            loss.item()
            * input_batch.size(0)
        )
        local_train_count += input_batch.size(0)

    scheduler.step()


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()

    local_test_loss = 0.0
    local_test_count = 0

    with torch.no_grad():

        for (
            input_batch,
            target_batch,
        ) in test_loader:

            input_batch = (
                input_batch.to(
                    device,
                    non_blocking=True,
                )
            )

            target_batch = (
                target_batch.to(
                    device,
                    non_blocking=True,
                )
            )

            prediction = model(
                input_batch
            )

            loss = criterion(
                prediction,
                target_batch,
            )

            local_test_loss += (
                loss.item()
                * input_batch.size(0)
            )
            local_test_count += input_batch.size(0)

    statistics = torch.tensor(
        [
            local_train_loss,
            local_train_count,
            local_test_loss,
            local_test_count,
        ],
        dtype=torch.float64,
        device=device,
    )

    if distributed:
        dist.all_reduce(
            statistics,
            op=dist.ReduceOp.SUM,
        )

    train_loss = (
        statistics[0].item()
        / statistics[1].item()
    )
    test_loss = (
        statistics[2].item()
        / statistics[3].item()
    )

    if rank == 0 and (
        epoch == 0
        or (epoch + 1) % 5 == 0
    ):

        print(
            f"Epoch "
            f"{epoch + 1:4d} "
            f"| train "
            f"{train_loss:.6e} "
            f"| test "
            f"{test_loss:.6e}"
        )


# ============================================================
# Save model
# ============================================================

if distributed:
    dist.barrier()

if rank == 0:

    inference_model = (
        model.module
        if distributed
        else model
    )

    os.makedirs(
        "models",
        exist_ok=True,
    )
    os.makedirs(
        "fig",
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                inference_model.state_dict(),

            "mean":
                float(mean),

            "std":
                float(std),

            "nx":
                nx,

            "ny":
                ny,

            "kappa":
                kappa,

            "dt":
                dt,

            "n_steps":
                n_steps,
        },
        "models/heat2d_fno.pt",
    )

    print(
        "Saved models/heat2d_fno.pt"
    )

    # ========================================================
    # Prediction example
    # ========================================================

    inference_model.eval()

    sample_input = torch.from_numpy(
        X_test_normalized[0:1]
    ).to(device)

    with torch.no_grad():

        sample_prediction = inference_model(
            sample_input
        )[0].cpu().numpy()

    sample_prediction = (
        sample_prediction * std
        + mean
    )

    initial = X_test[0]
    truth = Y_test[0]

    # ========================================================
    # Relative L2 error
    # ========================================================

    relative_l2 = (
        np.linalg.norm(
            sample_prediction - truth
        )
        /
        np.linalg.norm(truth)
    )

    print(
        "Example relative L2 error:",
        relative_l2,
    )

    # ========================================================
    # Plot
    # ========================================================

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    images = [
        initial,
        truth,
        sample_prediction,
        np.abs(
            sample_prediction - truth
        ),
    ]

    titles = [
        "Initial condition",
        "Numerical solution",
        "FNO prediction",
        "Absolute error",
    ]

    for ax, image, title in zip(
        axes,
        images,
        titles,
    ):

        im = ax.imshow(
            image.T,
            origin="lower",
            extent=(0, 1, 0, 1),
            aspect="equal",
        )

        ax.set_title(title)

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        fig.colorbar(
            im,
            ax=ax,
            shrink=0.8,
        )

    plt.tight_layout()

    plt.savefig(
        "fig/heat2d_fno_result.png",
        dpi=150,
    )

    plt.close(fig)

    print(
        "Saved fig/heat2d_fno_result.png"
    )

if distributed:
    dist.barrier()
    dist.destroy_process_group()
