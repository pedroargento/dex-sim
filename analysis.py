import numpy as np

from data_structures import SimulationResults


def summarize_df_requirement(sim: SimulationResults) -> None:
    aes = sim.aes_df_required
    fxd = sim.fxd_df_required

    print("\n=== DF REQUIREMENT SUMMARY (AES vs FXD) ===")
    print(f"AES DF required mean:      {np.mean(aes):,.0f}")
    print(f"AES DF required 95th pct:  {np.percentile(aes, 95):,.0f}")
    print(f"AES DF required 99th pct:  {np.percentile(aes, 99):,.0f}")

    print(f"FXD DF required mean:      {np.mean(fxd):,.0f}")
    print(f"FXD DF required 95th pct:  {np.percentile(fxd, 95):,.0f}")
    print(f"FXD DF required 99th pct:  {np.percentile(fxd, 99):,.0f}")


def summarize_defaults_and_haircuts(sim: SimulationResults) -> None:
    aes_def_rate = np.mean(sim.aes_defaults)
    fxd_def_rate = np.mean(sim.fxd_defaults)

    aes_df_exhaust = np.mean(sim.aes_df_exhausted)
    fxd_df_exhaust = np.mean(sim.fxd_df_exhausted)

    print("\n=== DEFAULTS & DF EXHAUSTION ===")
    print(f"AES default rate (paths):        {aes_def_rate:.2%}")
    print(f"AES paths w/ DF exhausted:       {aes_df_exhaust:.2%}")

    print(f"FXD default rate (paths):        {fxd_def_rate:.2%}")
    print(f"FXD paths w/ DF exhausted:       {fxd_df_exhaust:.2%}")

    # Haircuts
    aes_hc = sim.aes_haircuts
    fxd_hc = sim.fxd_haircuts

    print("\n=== HAIRCUTS (fraction of VM not paid) ===")
    if np.any(aes_hc > 0):
        nonzero = aes_hc[aes_hc > 0]
        print(f"AES median haircut:              {np.percentile(nonzero, 50):.2%}")
        print(f"AES 95th pct haircut:            {np.percentile(nonzero, 95):.2%}")
    else:
        print("AES: No haircuts observed.")

    if np.any(fxd_hc > 0):
        nonzero = fxd_hc[fxd_hc > 0]
        print(f"FXD median haircut:              {np.percentile(nonzero, 50):.2%}")
        print(f"FXD 95th pct haircut:            {np.percentile(nonzero, 95):.2%}")
    else:
        print("FXD: No haircuts observed.")
