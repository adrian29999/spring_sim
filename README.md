# Spring Simulation With PID

An interactive mass-spring-damper simulator with an optional PID controller, built with NumPy and Matplotlib. Adjust the physical parameters and PID gains with sliders and watch the response, phase portrait and numerical-error diagnostics update live.

## Features

- Mass-spring-damper model: `m·x'' + c·x' + k·x = F`
- Optional PID controller (Kp, Ki, Kd) with setpoint support
- RK4 time integration
- Live plots: position and velocity vs time, phase portrait, RK4 error vs exact solution, and formula-vs-scan fast-zeta comparison
- Two settling-time markers: a fixed 0.1% band and an adjustable "band 2"
- **Crit** and **fast ζ** snap buttons for `m`, `k`, `c`, `Kp` and `Kd`
- Info panel with the equation of motion, effective constants, state-space matrices, eigenvalues and damping classification

## Requirements

- Python 3.9+
- NumPy
- Matplotlib (with an interactive backend, e.g. TkAgg or Qt)

## Installation

```bash
git clone https://github.com/299999/spring_sim.git
cd spring_sim
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python spring_pid_simulation.py
```

### Controls

| Control | Description |
| --- | --- |
| Sliders | Mass, stiffness, damping, initial position/velocity, RK4 step, band 2, Kp, Ki, Kd |
| `Crit` buttons | Snap the variable so the damping ratio ζ = 1 |
| `ζ=…` buttons | Snap the variable to the fastest ζ for the band 2 threshold |
| enable PID | Toggle the PID controller |
| Reset | Restore all defaults |
| t_end | Toggle the simulation length between 15 s and 5 s |

Button colours: **red** means the target is outside the slider range, **khaki** means the fast ζ came from the scan (approximate).

## Documentation

All notes on variables, formulas, the fast-zeta method, validation graphs and known limitations are in the large docstring at the top of [`spring_pid_simulation.py`](spring_pid_simulation.py).

Key limitations to be aware of:

- Ki is only first-order accurate in the simulation (it is not an RK4 state).
- RK4 can blow up when `|λ_max|·dt` is too large (e.g. `m=1, c=15, Kd=20, dt=0.1`).
- The PID is ideal: no saturation, delay, noise or anti-windup.
- One degree of freedom, linear viscous friction only, no gravity.

## Project structure

```
spring_sim/
├── spring_pid_simulation.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## License

Released under the [MIT License](LICENSE). Replace the placeholder name in `LICENSE` with your own.
