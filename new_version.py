import lib.tf_silent
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from lib.pinn import PINN
from lib.network import Network
from lib.optimizer import L_BFGS_B

from params import LinearHomo1D

class InitialConditionLoader:
    """
    Load initial conditions from k-Wave .mat file.
    """

    def __init__(self, params):
        self.params = params

        with h5py.File(
            f"data/1d_kWave_BonA_{round(self.params.BonA)}_Nt_{self.params.Nt}_Nx_{self.params.Nx}_local.mat",
            "r",
        ) as f:

            self.p_fd = np.array(f["sensor_data"]["p"]).T
            self.u_fd = np.array(f["sensor_data"]["ux"]).T
            self.rho_fd = np.array(f["sensor_data"]["rhox"]).T

        assert self.p_fd.shape == (
            self.params.Nx,
            self.params.Nt,
        ), f"Unexpected p_fd shape: {self.p_fd.shape}"

        assert self.u_fd.shape == (
            self.params.Nx,
            self.params.Nt,
        ), f"Unexpected u_fd shape: {self.u_fd.shape}"

        assert self.rho_fd.shape == (
            self.params.Nx,
            self.params.Nt,
        ), f"Unexpected rho_fd shape: {self.rho_fd.shape}"

    def get_mid_pressure(self):
        """
        Use the middle time snapshot as the initial condition.

        Returns:
            p_mid : (Nx, 1)
        """

        mid_idx = self.params.Nt // 2

        # full spatial field at middle time
        p_mid = self.p_fd[:, mid_idx:mid_idx + 1]

        return p_mid


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    params = LinearHomo1D()

    # ------------------------------------------------------------------
    # Load initial-condition data
    # ------------------------------------------------------------------

    ic_loader = InitialConditionLoader(params)

    p_ini = ic_loader.get_mid_pressure()
    plt.plot(p_ini)

    print("Initial pressure shape :", p_ini.shape)

    # ------------------------------------------------------------------
    # Build neural network
    # ------------------------------------------------------------------

    network = Network.build()
    network.summary()

    # ------------------------------------------------------------------
    # Build PINN
    # ------------------------------------------------------------------

    pinn = PINN(network).build()

    # ------------------------------------------------------------------
    # Generate PDE collocation points
    # ------------------------------------------------------------------

    # number of training samples
    num_train_samples = 10000
    # number of test samples
    num_test_samples = 1000

    tx_eqn = np.random.rand(num_train_samples, 2)

    # t domain
    tx_eqn[:, 0] = params.t_end * tx_eqn[:, 0]

    # x domain
    tx_eqn[:, 1] = (
        (params.x_end) * tx_eqn[:, 1]
    )

    # ------------------------------------------------------------------
    # Initial-condition training points
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Initial-condition training points (UPSCALE TO 10,000)
    # ------------------------------------------------------------------
    # Create the complete grid coordinates
    x_ic_full = np.linspace(0, params.x_end, params.Nx)
    
    # Randomly select 10,000 indices from your 1,520 points (with replacement)
    # This scales your data to 10,000 samples while keeping the physics intact!
    rand_idx = np.random.choice(params.Nx, size=num_train_samples, replace=True)
    
    x_ic = x_ic_full[rand_idx]
    t_ic = np.zeros_like(x_ic)
    tx_ini = np.stack([t_ic, x_ic], axis=-1)

    # Match the upsampled indices for initial pressure
    u_ini = p_ini[rand_idx].astype(np.float32)
    du_dt_ini = np.zeros_like(u_ini) 

    # ------------------------------------------------------------------
    # Dummy Boundary Setup (10,000 rows)
    # ------------------------------------------------------------------
    # Neutral dummy points
    tx_bnd = np.zeros((num_train_samples, 2), dtype=np.float32)
    u_bnd_neutral = network.predict(tx_bnd, batch_size=num_test_samples).astype(np.float32)

    # PDE residual target 
    u_zero_eqn = np.zeros((num_train_samples, 1), dtype=np.float32)

    # ------------------------------------------------------------------
    # Train PINN
    # ------------------------------------------------------------------
    # Every single array here is now perfectly sized at (10000, 2)
    x_train = [
        tx_eqn,
        tx_ini,
        tx_bnd,
    ]

    # Every single array here is now perfectly sized at (10000, 1)
    y_train = [
        u_zero_eqn,    # 1. PDE residual
        u_ini,         # 2. Upsampled Initial condition (u)
        du_dt_ini,     # 3. Upsampled Initial velocity (du/dt)
        u_bnd_neutral, # 4. Neutral Dummy Target (Free Boundary)
    ]

    lbfgs = L_BFGS_B(
        model=pinn,
        x_train=x_train,
        y_train=y_train,
    )

    lbfgs.fit()

    # ------------------------------------------------------------------
    # Predict solution
    # ------------------------------------------------------------------

    t_flat = np.linspace(
        0,
        params.t_end,
        num_test_samples,
    )

    x_flat = np.linspace(
        0,
        params.x_end,
        num_test_samples,
    )

    t, x = np.meshgrid(t_flat, x_flat)

    tx = np.stack(
        [t.flatten(), x.flatten()],
        axis=-1,
    )

    u_pred = network.predict(
        tx,
        batch_size=num_test_samples,
    )

    u_pred = u_pred.reshape(t.shape)

    # ------------------------------------------------------------------
    # Plot predicted field
    # ------------------------------------------------------------------

    fig = plt.figure(figsize=(8, 5))

    gs = GridSpec(2, 3)

    plt.subplot(gs[0, :])

    vmin = np.min(u_pred)
    vmax = np.max(u_pred)

    plt.pcolormesh(
        t,
        x,
        u_pred,
        cmap="rainbow",
        norm=Normalize(vmin=vmin, vmax=vmax),
        shading="auto",
    )

    plt.xlabel("t")
    plt.ylabel("x")

    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label("u(t,x)")

    # ------------------------------------------------------------------
    # Cross sections
    # ------------------------------------------------------------------

    t_cross_sections = [1.0, 2.0, 3.0]

    for i, t_cs in enumerate(t_cross_sections):

        plt.subplot(gs[1, i])

        tx_cs = np.stack(
            [
                np.full(x_flat.shape, t_cs),
                x_flat,
            ],
            axis=-1,
        )

        u_cs = network.predict(
            tx_cs,
            batch_size=num_test_samples,
        )

        plt.plot(x_flat, u_cs)

        plt.title(f"t = {t_cs}")
        plt.xlabel("x")
        plt.ylabel("u(t,x)")

    plt.tight_layout()

    plt.savefig(
        "result_img_free_boundary.png",
        transparent=True,
        dpi=300,
    )

    plt.show()