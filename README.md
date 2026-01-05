## Neural Control Contraction Metric (CCM)
`Please note that this is ongoing work. I will likely add in LQR baselines for improved simulation results.`

This repo explores **neural Control Contraction Metrics (CCMs)** for a nonlinear, underactuated **cart–pendulum system**.

- Control-affine dynamics: (`ẋ = f(x) + B(x)u`)
- **Sin/cos state embedding** to avoid angle discontinuities
- Learns a **state-dependent contraction metric matrix** \(M(x)\) and **differential gain matrix** \(K(x)\)
- Enforces the continuous-time CCM condition on the closed-loop *differential* dynamics

This guarantees **incremental (relative) exponential stability**: nearby trajectories contract toward each other, as defined by a CCM-loop.

### Structure
- python/train_ccm_metric.py -->  CCM training
- matlab/cartpend_ccm_demo.m --> Nonlinear simulation + plots


