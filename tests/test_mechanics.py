"""Sanity checks against numbers quoted in the paper.  Run with:  python -m pytest tests"""
import numpy as np
import pytest

from rl_growth.mechanics import ThinWallArtery, BilayerAorta
from rl_growth.units import mmhg_to_kpa, wall_shear_stress
from rl_growth.envs import IAAGrowthEnv, BilayerGrowthEnv, make_action_table, log_reward
from rl_growth.rl import policy_iteration


def test_wall_shear_stress_homeostatic_values():
    assert wall_shear_stress(2.8, 4.5, 0.464) == pytest.approx(26.8, abs=0.1)      # Table 2
    assert wall_shear_stress(856.5, 4.5, 7.99) == pytest.approx(1.6035, abs=0.002)  # Table 5


def test_thin_wall_day0_state():
    art = ThinWallArtery()
    out = art.solve(np.ones((1, 3)), mmhg_to_kpa(111.6))
    assert out["r_i"][0] == pytest.approx(0.418, abs=0.01)          # Fig. 6(c) day 0
    assert out["sig_tt"][0] == pytest.approx(194.0, abs=6.0)       # Fig. 8(a) day 0
    assert out["sig_zz"][0] == pytest.approx(258.0, abs=8.0)       # Fig. 8(b) day 0
    # Laplace equation satisfied
    assert out["sig_tt"][0] == pytest.approx(mmhg_to_kpa(111.6) * out["r_i"][0] / out["h"][0], rel=1e-8)
    # elastic incompressibility  lam_r lam_t lam_z = det(Fg)
    assert out["lam_r"][0] * out["lam_theta"][0] * art.lam_z == pytest.approx(1.0, rel=1e-10)


def test_bilayer_before_growth_matches_table5():
    ao = BilayerAorta()
    o = ao.loaded(ao.identity_growth())
    assert o["r_i"][0] == pytest.approx(7.99, abs=0.01)
    m_tt, m_zz = ao.layer_mean(o["sig_tt"])[0], ao.layer_mean(o["sig_zz"])[0]
    assert m_tt[0] == pytest.approx(160.10, rel=0.01)
    assert m_zz[0] == pytest.approx(101.87, rel=0.01)
    assert m_tt[1] == pytest.approx(55.96, rel=0.01)
    assert m_zz[1] == pytest.approx(36.70, rel=0.01)
    assert abs(o["residual_p"][0]) < 1e-9
    assert -ao.p_kPa < o["sig_rr"][0, 0, 0] < 0.0          # radial stress between -p (lumen) and 0 (outer wall)


def test_bilayer_unloaded_identity_is_stress_free():
    ao = BilayerAorta()
    lam_g = ao.identity_growth()
    u = ao.unloaded(lam_g)
    assert u["r_i"] == pytest.approx(ao.layers[0].R_i, abs=1e-6)
    assert np.abs(u["sig_tt"]).max() < 1e-8
    assert ao.opening_angle(lam_g, 0)["alpha_deg"] == pytest.approx(0.0, abs=1e-6)


def test_bilayer_growth_generates_residual_stress():
    ao = BilayerAorta()
    lam_g = ao.identity_growth()
    lam_g[0, 0, :, 1] = np.linspace(1.03, 0.97, 5)      # media grows more at the inner wall
    u = ao.unloaded(lam_g)
    assert u["sig_tt"][0, 0, 0] < 0 < u["sig_tt"][0, 0, -1]   # compressive inside, tensile outside
    assert ao.opening_angle(lam_g, 0)["alpha_deg"] > 5.0


def test_actions_and_reward():
    assert make_action_table(("r", "t", "z"), 0.01).shape == (27, 3)
    assert make_action_table(("t", "z"), 0.01).shape == (9, 3)
    assert log_reward(np.exp(150.0 / 40.0)) == pytest.approx(0.0)
    assert log_reward(1e-9) == 80.0 and log_reward(1e9) == -80.0


def test_iaa_env_grid_and_dp():
    env = IAAGrowthEnv(lam_g_range=(0.95, 1.05))         # small grid (11^3) for speed
    lam, s, r, nxt = env.build_mdp()
    assert lam.shape == (1331, 3) and nxt.shape == (1331, 27)
    pol, V = policy_iteration(nxt, r, gamma=0.99, verbose=False, preferred_action=env.no_growth_action)
    # the best state should be a fixed point of the policy
    best = int(np.argmax(r))
    assert nxt[best, pol[best]] == best


def test_bilayer_env_step():
    env = BilayerGrowthEnv(layer=0)
    s = env.reset(4)
    assert s.shape == (4, 3)
    s2, rew, done, info = env.step(np.zeros(4, int))
    assert s2.shape == (4, 3) and rew.shape == (4,) and done.shape == (4,)
