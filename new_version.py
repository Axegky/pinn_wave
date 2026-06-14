import random
import lib.tf_silent
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from lib.pinn_new import PINN
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

    def get_p_ini(self, time_idx=None):
        """
        Use the middle time snapshot as the initial condition.

        Returns:
            p_mid : (Nx, 1)
        """

        if time_idx is None:
            time_idx = self.params.Nt // 2

        # full spatial field at middle time
        p_mid = self.p_fd[:, time_idx:time_idx + 1]

        return p_mid
    
    def get_dpdt(self, time_idx=None):
        """
        Compute the time derivative of pressure at the initial time.

        Returns:
            dpdt_ini : (Nx, 1)
        """

        dt = self.params.dt
        if time_idx is None:
            time_idx = self.params.Nt // 2

        dpdt_ini = (self.p_fd[:, time_idx + 1:time_idx + 2] - self.p_fd[:, time_idx:time_idx + 1]) / dt

        return dpdt_ini


if __name__ == "__main__":
    random_seed_name = 123
    random.seed(random_seed_name)
    np.random.seed(random_seed_name)
    tf.random.set_seed(random_seed_name)

    params = LinearHomo1D()
    ic_loader = InitialConditionLoader(params)
    time_idx = params.Nt // 2

    p_ini = ic_loader.get_p_ini(time_idx)

    print("p_ini:")
    print(" min =", p_ini.min())
    print(" max =", p_ini.max())

    dpdt = ic_loader.get_dpdt(time_idx)

    print("dpdt:")
    print(" min =", dpdt.min())
    print(" max =", dpdt.max())

    u_scale = np.max(np.abs(p_ini))
    v_scale = np.max(np.abs(dpdt))
    t_scale = params.t_end
    x_scale = params.x_end

    p_ini_scaled = p_ini / u_scale
    dpdt_scaled = dpdt / (u_scale / t_scale)

    # number of training samples
    num_train_samples = 10000
    # number of test samples
    num_test_samples = 1000

    network = Network.build()
    network.summary()
    # build a PINN model
    pinn = PINN(network, params.c0, t_scale, x_scale).build()

    print("PINN inputs:", len(pinn.inputs))
    print(pinn.inputs)

    # designate Adam optimiser and loss function for training
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-3
    )

    pinn.compile(
        optimizer=optimizer,
        loss="mse"
    )

    # create training input
    tx_eqn = np.random.rand(num_train_samples, 2)

    # Create the complete grid coordinates
    x_ic_full = np.linspace(0, params.x_end, params.Nx)
    
    rand_idx = np.random.choice(params.Nx, size=num_train_samples, replace=True)
    
    x_ic = x_ic_full[rand_idx]
    t_ic = np.zeros_like(x_ic)
    tx_ini = np.stack([t_ic, x_ic], axis=-1)
    # scale for t and x in initial condition to [0,1]
    tx_ini = np.stack([
        np.zeros_like(x_ic),
        x_ic / x_scale
    ], axis=-1)

    u_ini = (p_ini[rand_idx] / u_scale).astype(np.float32)
    du_dt_ini = (ic_loader.get_dpdt()[rand_idx] / (u_scale / t_scale)).astype(np.float32)
    
    # PDE residual target 
    u_zero_eqn = np.zeros((num_train_samples, 1), dtype=np.float32)


    # Every single array here is now perfectly sized at (10000, 2)
    x_train = [
        tx_eqn,
        tx_ini
    ]

    # Every single array here is now perfectly sized at (10000, 1)
    y_train = [
        u_zero_eqn,    
        u_ini,         
        du_dt_ini    
    ]

    history = pinn.fit(
        x=x_train,
        y=y_train,
        epochs=5000,
        batch_size=num_train_samples,  # full-batch Adam
        shuffle=False,                 # keep deterministic
        verbose=1
    )

    # lbfgs = L_BFGS_B(
    #     model=pinn,
    #     x_train=x_train,
    #     y_train=y_train,
    # )

    # lbfgs.fit()

    # Predict solution
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
        [
            t.flatten() / t_scale,
            x.flatten() / x_scale,
        ],
        axis=-1,
    )

    u_pred = network.predict(
        tx,
        batch_size=num_test_samples,
    )

    u_pred = u_pred.reshape(t.shape)
    u_pred = u_pred * u_scale

    # Plot predicted field
    fig = plt.figure(figsize=(8, 5))

    gs = GridSpec(2, 3)

    plt.subplot(gs[0, :])

    vmin = np.min(u_pred)
    vmax = np.max(u_pred)

    plt.pcolormesh(
        t,
        x,
        u_pred,
        cmap="coolwarm",
        norm=Normalize(vmin=vmin, vmax=vmax),
        shading="auto",
    )

    plt.xlabel("t")
    plt.ylabel("x")

    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label("u(t,x)")

    # Cross sections at different time snapshots
    t_cross_sections = [0, 0.5e-5, 1e-5]

    for i, t_cs in enumerate(t_cross_sections):

        plt.subplot(gs[1, i])

        tx_cs = np.stack(
            [
                np.full(x_flat.shape, t_cs / t_scale),
                x_flat / x_scale,
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
        f"result_img_free_boundary_t0_{time_idx}_seed_{random_seed_name}.png",
        transparent=True,
        dpi=300,
    )

    plt.show()