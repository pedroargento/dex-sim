import numpy as np
import matplotlib.pyplot as plt

from data_structures import SimulationResults


def plot_df_requirement_hist(
    sim: SimulationResults, filename: str = "df_required_hist.png"
) -> None:
    aes_df_req = sim.aes_df_required
    fxd_df_req = sim.fxd_df_required

    plt.figure(figsize=(10, 6))
    plt.hist(aes_df_req, bins=50, alpha=0.7, label="AES DF required")
    plt.hist(fxd_df_req, bins=50, alpha=0.7, label="FXD DF required")
    plt.xlabel("DF Required per Path (USD)")
    plt.ylabel("Frequency")
    plt.title("Default Fund Requirement Distribution: AES vs FXD")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def plot_sample_paths(sim: SimulationResults, max_paths: int = 5) -> None:
    price_paths = sim.price_paths
    lev_la = sim.leverage_long_aes
    lev_lf = sim.leverage_long_fxd
    R_aes = sim.R_aes
    R_fxd = sim.R_fxd
    br_aes = sim.breaker_aes
    br_fxd = sim.breaker_fxd

    num_sample_paths, horizon = price_paths.shape
    max_plots = min(max_paths, num_sample_paths)
    t_axis = np.arange(horizon)

    for k in range(max_plots):
        # Price path
        plt.figure(figsize=(10, 4))
        plt.plot(t_axis, price_paths[k])
        plt.xlabel("Time step")
        plt.ylabel("Price")
        plt.title(f"Sample path {k} - Price")
        plt.grid(True)
        fname = f"sample_{k}_price.png"
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")

        # Leverage (long side)
        plt.figure(figsize=(10, 4))
        plt.plot(t_axis, lev_la[k], label="AES long")
        plt.plot(t_axis, lev_lf[k], label="FXD long")
        plt.xlabel("Time step")
        plt.ylabel("Leverage (long)")
        plt.title(f"Sample path {k} - Long Leverage AES vs FXD")
        plt.legend()
        plt.grid(True)
        fname = f"sample_{k}_leverage_long.png"
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")

        # Risk factor R_t
        plt.figure(figsize=(10, 4))
        plt.plot(t_axis, R_aes[k], label="AES R_t")
        plt.plot(t_axis, R_fxd[k], label="FXD R_t")
        plt.xlabel("Time step")
        plt.ylabel("R_t")
        plt.title(f"Sample path {k} - Systemic Risk Factor R_t")
        plt.legend()
        plt.grid(True)
        fname = f"sample_{k}_Rt.png"
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")

        # Breaker state (step plot)
        plt.figure(figsize=(10, 4))
        plt.step(t_axis, br_aes[k], where="post", label="AES breaker")
        plt.step(t_axis, br_fxd[k], where="post", label="FXD breaker")
        plt.yticks([0, 1, 2], ["NORMAL", "SOFT", "HARD"])
        plt.xlabel("Time step")
        plt.ylabel("Breaker State")
        plt.title(f"Sample path {k} - Breaker State AES vs FXD")
        plt.legend()
        plt.grid(True)
        fname = f"sample_{k}_breaker.png"
        plt.savefig(fname)
        plt.close()
        print(f"Saved {fname}")
