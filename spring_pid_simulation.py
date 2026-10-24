import itertools
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button


SETTLE_FRAC = 1e-3   # fixed settling band, fraction of travel distance (0.1%)

p: dict[str, Any] = dict(m=1.0, k=10.0, c=0, x0=1.0, v0=0.0, dt=0.001, err_pct=5.0,
         setpoint=0.0, kp=0.0, ki=0.0, kd=0.0, pid_on=False)
t_end = 15
T_END_LONG = 15
T_END_SHORT = 5

tolerance = 1e-5

b_nthn = 'lightgray'
b_hover = 'skyblue'
b_na = 'red'
b_scan = 'khaki'   # fast zeta button colour when the zeta came from the scan

ZETA_CRIT = 1.0
_scan_cache: dict[tuple[float, ...], Any] = {}
#stores earlier results, so there is no need to recalculate if they repeat


def fast_zeta_formula(tol_frac):
    #fastest zeta only with a certain overshoot percent, formula method
    log_inv = -np.log(tol_frac)
    return float(log_inv / np.sqrt(np.pi**2 + log_inv**2))


def response_x(q, zeta, t):
    # handles position, 2nd order but no labels 
    k_eff, _ = effective(q)
    m = q['m']
    c_cand = 2.0 * zeta * np.sqrt(k_eff * m)
    mat = np.array([[0.0, 1.0],
                    [-k_eff / m, -c_cand / m]])
    eigvals, eigvecs = np.linalg.eig(mat)
    try:
        coef = np.linalg.solve(eigvecs, np.array([q['x0'], q['v0']]))
    except np.linalg.LinAlgError:
        return None
    with np.errstate(all='ignore'):
        return np.real(np.exp(np.outer(t, eigvals)) @ (coef * eigvecs[0, :]))


def fast_zeta_scan(q, tol_frac, t_stop):
    #fastest zeta only with a certain overshoot percent, scan method
    k_eff, _ = effective(q)
    m = q['m']
    x0, v0 = q['x0'], q['v0']
    cache_key = tuple(round(float(val), 6) for val in (m, k_eff, x0, v0, tol_frac, t_stop))
    if cache_key in _scan_cache:
        return _scan_cache[cache_key]
    band = tol_frac * (abs(x0) if x0 != 0 else 1.0)
    t = np.linspace(0.0, t_stop, 6001)
    best_z, best_ts = None, np.inf
    for z in np.append(np.arange(0.2, 1.0, 0.005), 1.0 - 1e-6):
        x = response_x(q, z, t)
        if x is None:
            continue
        out = np.nonzero(np.abs(x) > band)[0]
        if len(out) == 0:
            ts = 0.0
        elif out[-1] + 1 < len(t):
            ts = t[out[-1] + 1]
        else:
            continue   # never settles inside the window
        if ts < best_ts - 1e-12:
            best_z, best_ts = float(z), ts
    _scan_cache[cache_key] = best_z
    return best_z


def zeta_fast_info(q):
    #decides which method to use
    tol = q['err_pct'] / 100.0
    z_formula = fast_zeta_formula(tol)
    ki = q['ki'] if q['pid_on'] else 0.0
    moving = q['x0'] != 0 or q['v0'] != 0
    if moving and (ki > 0 or q['v0'] != 0):
        z_scan = fast_zeta_scan(q, tol, t_end)
        return (z_scan if z_scan is not None else z_formula), True
    return z_formula, False


def zeta_fast_for(q):
    #the current fast zeta
    return zeta_fast_info(q)[0]


def effective(q):
    # gets the effective stiffness and damping, if pid is on
    if q['pid_on']:
        return q['k'] + q['kp'], q['c'] + q['kd']
    return q['k'], q['c']


def equilibrium(q):
    # gets the rest position
    if not q['pid_on']:
        return 0.0
    if q['ki'] > 0:
        return q['setpoint']
    return q['kp'] * q['setpoint'] / (q['k'] + q['kp'])


def band_time(x_hist, t_hist, center, threshold):
    # time after which x stays within the threshold.
    # a blown up run (nan/inf) never settles, without this check abs(nan) > threshold
    # is False and the function would wrongly report t = 0
    if not np.all(np.isfinite(x_hist)):
        return None
    if abs(x_hist[-1] - center) > threshold:
        return None
    for idx in range(len(x_hist) - 1, -1, -1):
        if abs(x_hist[idx] - center) > threshold:
            return t_hist[idx + 1]
    return t_hist[0]


def root_error(m, c_eff, k_eff, ki, approx):
    # approx second order roots with ki ignored vs third order roots incl ki 
    true_roots = np.roots([m, c_eff, k_eff, ki])
    best_pair = min(itertools.permutations(true_roots, len(approx)),
                    key=lambda pair: sum(abs(a - r) for a, r in zip(approx, pair)))
    out = []
    for a, r in zip(approx, best_pair):
        e = abs(a - r)
        out.append((e, 100.0 * e / abs(a) if abs(a) > 0 else 0.0))
    return out


def zeta_and_wn(m, k, c):
    #computes zeta and natural angular frequency
    wn = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(k * m))
    return zeta, wn


def snap_targets(q, z):
    # the values each individual variable would need, keeping the rest the same
    # to have the damping ratio z
    k_eff, c_eff = effective(q)
    kp_e = q['kp'] if q['pid_on'] else 0.0
    kd_e = q['kd'] if q['pid_on'] else 0.0
    m = q['m']

    t_c = 2.0 * z * np.sqrt(max(k_eff * m, 0.0)) - kd_e
    t_k = (c_eff**2) / (4.0 * z**2 * m) - kp_e if m > 0 else float('inf')
    t_m = (c_eff**2) / (4.0 * z**2 * k_eff) if k_eff > 0 else float('inf')

    # always evaluated with the PID treated as enabled
    k_on, c_on = effective(dict(q, pid_on=True))
    t_kd = 2.0 * z * np.sqrt(max(k_on * m, 0.0)) - q['c']
    t_kp = (c_on**2) / (4.0 * z**2 * m) - q['k'] if m > 0 else float('inf')
    return {'m': t_m, 'k': t_k, 'c': t_c, 'kp': t_kp, 'kd': t_kd}


def eigen_info(m, k, c, z_fast=None, band_pct=None):
    # calculates the eigenvalues by hand, and gets the label
    zeta, wn = zeta_and_wn(m, k, c)
    if zeta == 0:
        l1, l2 = 1j * wn, -1j * wn
        return (l1, l2, zeta, wn, 'Undamped, sustained oscillation.')
    discriminant = zeta**2 - 1
    # critical check comes first, so zeta just below 1 (e.g. 0.9998 after a
    # slider-stepped snap) is labelled critical instead of underdamped
    if abs(discriminant) < tolerance:
        l1, l2 = -wn, -wn
        return (l1, l2, zeta, wn, 'Critically damped, no oscillation.')
    if z_fast is not None and zeta < 1 and abs(zeta - z_fast) < 0.005:
        wd = wn * np.sqrt(1 - zeta ** 2)
        l1, l2 = -zeta * wn + 1j * wd, -zeta * wn - 1j * wd
        overshoot = 100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2))
        return (l1, l2, zeta, wn, f'Underdamped, fast for {band_pct:.1f}% band ({overshoot:.1f}% OS).')
    if discriminant < 0:
        wd = wn * np.sqrt(1 - zeta ** 2)
        l1, l2 = -zeta * wn + 1j * wd, -zeta * wn - 1j * wd
        return (l1, l2, zeta, wn, 'Underdamped, decaying oscillation.')
    l1 = -zeta * wn + wn * np.sqrt(discriminant)
    l2 = -zeta * wn - wn * np.sqrt(discriminant)
    return (l1, l2, zeta, wn, 'Overdamped, slow non-oscillatory.')


def analytical_x(t, m, k, c, x0, v0):
    #handles position manually, but only 2nd order, used for rk4 error graph
    zeta, wn = zeta_and_wn(m, k, c)
    if zeta < 1:
        wd = wn * np.sqrt(1 - zeta ** 2)
        return np.exp(-zeta * wn * t) * (
            x0 * np.cos(wd * t) + ((v0 + zeta * wn * x0) / wd) * np.sin(wd * t))
    if abs(zeta - 1) < tolerance:
        return np.exp(-wn * t) * (x0 + (v0 + wn * x0) * t)
    discriminant = zeta**2 - 1
    l1 = -zeta * wn + wn * np.sqrt(discriminant)
    l2 = -zeta * wn - wn * np.sqrt(discriminant)
    C1 = (v0 - l2 * x0) / (l1 - l2)
    C2 = x0 - C1
    return C1 * np.exp(l1 * t) + C2 * np.exp(l2 * t)


def closed_loop_matrix(q):
    # makes the 2x2 matrix, using effective versions of variables
    m = q['m']
    return np.array([[0.0, 1.0],
                     [-(q['k'] + q['kp']) / m, -(q['c'] + q['kd']) / m]])


def reference_x(q, t_hist):
    # exact reference for the rk4 error graph; only exists with PID off
    if q['pid_on']:
        return None
    return analytical_x(t_hist, q['m'], q['k'], q['c'], q['x0'], q['v0'])



def derivative(state, params, integral_error):
    #calculates derivative of velocity and position
    x, v = state
    force = 0.0
    if params['pid_on']:
        error = params['setpoint'] - x
        force += params['kp'] * error + params['ki'] * integral_error + params['kd'] * (-v)
    acceleration = (force - params['c'] * v - params['k'] * x) / params['m']
    return np.array([v, acceleration])


def rk4_step(state, params, dt, integral_error):
    #fourth order runge-kutta, solves differential equations, step by step
    #NOTE: integral_error is a constant input here, it is not one of the RK4 states
    k1 = derivative(state, params, integral_error)
    k2 = derivative(state + dt / 2 * k1, params, integral_error)
    k3 = derivative(state + dt / 2 * k2, params, integral_error)
    k4 = derivative(state + dt * k3, params, integral_error)
    #weighted average, middle 2 count more
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def run_simulation(params, t_stop):
    state = np.array([params['x0'], params['v0']])
    integral_error = 0.0
    t = 0.0
    n_steps = int(round(t_stop / params['dt']))   # round, not int(), avoids float error dropping a step

    t_hist = np.zeros(n_steps + 1)
    x_hist = np.zeros(n_steps + 1)
    v_hist = np.zeros(n_steps + 1)

    x_hist[0] = params['x0']
    v_hist[0] = params['v0']

    for i in range(1, n_steps + 1):
        e = params['setpoint'] - state[0]

        # Ki limitation: integral_error is frozen for all 4 RK4 stages of this step,
        # then updated below with a rectangle rule (e at the START of the step).
        # So the Ki term is only first-order accurate, even though x and v use RK4.
        # With Ki = 0 or PID off this has no effect (see note 8).
        state = rk4_step(state, params, params['dt'], integral_error)
        integral_error += e * params['dt']
        t += params['dt']

        t_hist[i] = t
        x_hist[i] = state[0]
        v_hist[i] = state[1]

    ref = reference_x(params, t_hist)
    if ref is None:
        rk4_error_hist = None
    else:
        with np.errstate(all='ignore'):
            rk4_error_hist = x_hist - ref
    return t_hist, x_hist, v_hist, rk4_error_hist


fig = plt.figure(figsize=(14, 7))
fig.suptitle('Spring Simulation With PID', fontsize=22, x=0.5, ha='center')

slider_specs = [
    ('m',        1, 10,  p['m'],        'mass m (kg)'),
    ('k',        0.05, 35,  p['k'],        'stiffness k (N/m)'),
    ('c',        0.0, 15,  p['c'],        'damping c (N·s/m)'),
    ('x0',      -1.0, 1.0, p['x0'],       'initial x0 (m)'),
    ('v0',      -2.0, 2.0, p['v0'],       'initial v0 (m/s)'),
    ('dt',      0.001, 0.1, p['dt'],     'RK4 step dt (s)'),
    ('err_pct',  0.5, 20.0, p['err_pct'], 'band 2 (% travel)'),   # min must stay > SETTLE_FRAC*100
    ('kp',       0.0, 50,  p['kp'],       'Kp (N/m)'),
    ('ki',       0.0, 10,  p['ki'],       'Ki (N/(m·s))'),
    ('kd',       0.0, 20,  p['kd'],       'Kd (N·s/m)'),
]

sliders = {}
y = 0.88
for slider_key, min_val, max_val, start, slider_label in slider_specs:
    ax_s = fig.add_axes((0.12, y, 0.205, 0.025))
    sliders[slider_key] = Slider(ax_s, slider_label, min_val, max_val, valinit=start, valstep=0.001)
    sliders[slider_key].label.set_fontsize(8)
    y -= 0.038
y -= 0.05


def current_params():
    # the p dict updater
    q = dict(p)
    for key in sliders:
        q[key] = sliders[key].val
    return q


def make_snap_cb(target_key, z_func):
    # snaps to make the zeta what it needs to be
    def on_snap_click(_event):
        q = current_params()
        targets = snap_targets(q, z_func(q))
        val = np.clip(targets[target_key], sliders[target_key].valmin, sliders[target_key].valmax)
        sliders[target_key].set_val(val)
    return on_snap_click


SNAP_KEYS = ['m', 'k', 'c', 'kp', 'kd']   # no Ki: see notes

crit_snap_buttons = {}
for snap_key in SNAP_KEYS:
    pos = sliders[snap_key].ax.get_position()
    ax_crit_snap = fig.add_axes((pos.x1 + 0.05, pos.y0, 0.03, pos.height))
    crit_btn = Button(ax_crit_snap, 'Crit', color=b_nthn, hovercolor=b_hover)
    crit_btn.on_clicked(make_snap_cb(snap_key, lambda q: ZETA_CRIT))
    crit_snap_buttons[snap_key] = crit_btn

# fast zeta for the current band: approximate when PID is on (third Ki root ignored)
fast_snap_buttons = {}
for snap_key in SNAP_KEYS:
    pos = sliders[snap_key].ax.get_position()
    ax_fast_snap = fig.add_axes((pos.x1 + 0.09, pos.y0, 0.03, pos.height))
    fast_btn = Button(ax_fast_snap, f'ζ={zeta_fast_for(p):.2f}', color=b_nthn, hovercolor=b_hover)
    fast_btn.label.set_fontsize(7)
    fast_btn.on_clicked(make_snap_cb(snap_key, zeta_fast_for))
    fast_snap_buttons[snap_key] = fast_btn


ax_chk = fig.add_axes((0.05, y, 0.12, 0.05))
chk = CheckButtons(ax_chk, ['enable PID'], [p['pid_on']])

ax_reset = fig.add_axes((0.20, y, 0.12, 0.05))
btn_reset = Button(ax_reset, 'Reset')

ax_tend = fig.add_axes((0.35, y, 0.11, 0.05))
btn_tend = Button(ax_tend, f't_end: {t_end}', color=b_nthn, hovercolor=b_hover)

y -= 0.05

# Two text columns: model on the left, diagnostics on the right
ax_info = fig.add_axes((0.0375, 0.0, 0.375, y + 0.025))
ax_info.axis('off')
info_a = ax_info.text(0, 1, '', va='top', ha='left', family='monospace', fontsize=8)
info_b = ax_info.text(0.55, 1, '', va='top', ha='left', family='monospace', fontsize=8)

LABEL_SETTLE = f'Settling time ({SETTLE_FRAC * 100:g}%)'

ax_ts = fig.add_axes((0.5, 0.4, 0.45, 0.5))
ax_ts.set_title('Position & Velocity vs Time', fontsize=9)
line_v, = ax_ts.plot([], [], label='v (m/s)', color='skyblue')
line_x, = ax_ts.plot([], [], label='x (m)', color='black')
line_settle = ax_ts.axvline(0, color='purple', linestyle='--', linewidth=2.5, alpha=0.7, label=LABEL_SETTLE)
line_error_band = ax_ts.axvline(0, color='crimson', linestyle=':', linewidth=2.5, alpha=0.7,
                                label=f'Settling time ({p["err_pct"]:.1f}%)')
ax_ts.legend(fontsize=7, loc='upper right')
oob_text = ax_ts.text(0.02, 0.97, '', transform=ax_ts.transAxes, va='top', ha='left',
                      fontsize=8, color='crimson')
ax_ts.set_xlabel('t (s)')
ax_ts.set_xlim(0, t_end)
ax_ts.set_ylim(-8, 8)

ax_phase = fig.add_axes((0.5, 0.075, 0.1, 0.2))
ax_phase.set_title('Phase Portrait', fontsize=9)
line_phase, = ax_phase.plot([], [])
ax_phase.set_xlabel('x (m)')
ax_phase.set_ylabel('v (m/s)')
ax_phase.set_xlim(-5, 5)
ax_phase.set_ylim(-5, 5)

ax_rk4_error = fig.add_axes((0.675, 0.075, 0.1, 0.2))
ax_rk4_error.set_title('RK4 Error vs Exact', fontsize=9)
line_rk4_error, = ax_rk4_error.plot([], [])
ax_rk4_error.set_xlabel('t (s)')
ax_rk4_error.set_ylabel('error (m)')
ax_rk4_error.set_xlim(0, t_end)
ax_rk4_error.set_ylim(-1, 1)
rk4_na_text = ax_rk4_error.text(0.5, 0.5, '', transform=ax_rk4_error.transAxes,
                                ha='center', va='center', fontsize=9)

ax_zeta_error = fig.add_axes((0.85, 0.075, 0.1, 0.2))
ax_zeta_error.set_title('z Formula vs Scan', fontsize=9)
line_zeta_error, = ax_zeta_error.plot([], [], color='darkorange')
ax_zeta_error.set_xlabel('t (s)')
ax_zeta_error.set_ylabel('x(z formula) - x(z scan) (m)', fontsize=7)
ax_zeta_error.set_xlim(0, t_end)
ax_zeta_error.set_ylim(-1, 1)



def fit_y(ax, values):
    # linear axis scaling
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        ax.set_ylim(-1, 1)
        return
    lo, hi = float(values.min()), float(values.max())
    pad = 0.05 * (hi - lo) if hi > lo else max(abs(hi) * 0.05, 1e-12)
    ax.set_ylim(lo - pad, hi + pad)


def fit_phase(ax, x_vals, v_vals):
    # phase portrait: x and v get the SAME symmetric span, so both scales are equal
    vals = np.concatenate([np.asarray(x_vals, dtype=float), np.asarray(v_vals, dtype=float)])
    vals = vals[np.isfinite(vals)]
    lim = float(np.abs(vals).max()) * 1.05 if len(vals) > 0 else 0.0
    if lim <= 0:
        lim = 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


def fmt(z):
    #prettyprinting complex numbers
    if isinstance(z, complex):
        sign = '+' if z.imag >= 0 else '-'
        return f'{z.real:.3f} {sign} {abs(z.imag):.3f}j'
    return f'{z:.3f}'


def update_snap_colors(buttons, targets, ok_color=b_nthn):
    # different colors for buttons
    for name, target_val in targets.items():
        if name in buttons:
            s_obj = sliders[name]
            bad = (not np.isfinite(target_val)) or target_val < s_obj.valmin or target_val > s_obj.valmax
            col = b_na if bad else ok_color
            buttons[name].ax.set_facecolor(col)
            buttons[name].color = col


def redraw(_=None):
    #all the graphs are plotted and updated here
    for key in sliders:
        p[key] = sliders[key].val

    k_eff, c_eff = effective(p)

    t_hist, x_hist, v_hist, rk4_error_hist = run_simulation(p, t_end)
    line_x.set_data(t_hist, x_hist)
    line_v.set_data(t_hist, v_hist)
    line_phase.set_data(x_hist, v_hist)
    #gets the bigger value of x and v, refits y axis to that max
    max_of_xy = np.concatenate([x_hist, v_hist])
    fit_y(ax_ts, max_of_xy)
    # phase portrait: same span on both axes
    fit_phase(ax_phase, x_hist, v_hist)

    # rk4 error graph: only has an exact reference when PID is off
    if rk4_error_hist is None:
        line_rk4_error.set_data([], [])
        ax_rk4_error.set_ylim(-1, 1)
        rk4_na_text.set_text('n/a (PID on)')
    else:
        line_rk4_error.set_data(t_hist, rk4_error_hist)
        fit_y(ax_rk4_error, rk4_error_hist)
        rk4_na_text.set_text('')

    # response difference between using the formula's zeta and the scan's zeta
    # reruns always
    z_formula = fast_zeta_formula(p['err_pct'] / 100.0)
    z_scan = fast_zeta_scan(p, p['err_pct'] / 100.0, t_end)
    x_formula = response_x(p, z_formula, t_hist)
    x_scan = response_x(p, z_scan, t_hist) if z_scan is not None else None
    if x_formula is None or x_scan is None:
        line_zeta_error.set_data([], [])
        fit_y(ax_zeta_error, [])
    else:
        zeta_diff = x_formula - x_scan
        line_zeta_error.set_data(t_hist, zeta_diff)
        fit_y(ax_zeta_error, zeta_diff)
    
    
    settle_time = error_band_time = None   # out of bounds off the graph
    if c_eff > 0:
        final_destination = equilibrium(p)
        travel = abs(p['x0'] - final_destination)
        scale = travel if travel > 0 else 1.0

        settle_time = band_time(x_hist, t_hist, final_destination, SETTLE_FRAC * scale)
        error_band_time = band_time(x_hist, t_hist, final_destination, p['err_pct'] / 100.0 * scale)

    name_error = f'Settling time ({p["err_pct"]:.1f}%)'
    shown = [line_x, line_v]
    out_of_bounds = []
    for line, t_val, name in ((line_settle, settle_time, LABEL_SETTLE),
                              (line_error_band, error_band_time, name_error)):
        if t_val is None:
            line.set_visible(False)
            out_of_bounds.append(f'{name}: out of bounds')
        else:
            line.set_xdata([t_val])
            line.set_visible(True)
            shown.append(line)

    line_error_band.set_label(name_error)
    ax_ts.legend(handles=shown, fontsize=7, loc='upper right')
    oob_text.set_text('\n'.join(out_of_bounds))

    crit_t = snap_targets(p, ZETA_CRIT)
    z_fast, used_scan = zeta_fast_info(p)
    fast_t = snap_targets(p, z_fast)
    for fb in fast_snap_buttons.values():
        fb.label.set_text(f'ζ={z_fast:.2f}')

    l1, l2, zeta, wn, roots = eigen_info(p['m'], k_eff, c_eff, z_fast, p['err_pct'])

    if rk4_error_hist is None:
        val_text = '  max|err| = n/a\n  MAE      = n/a\n'
    else:
        abs_error = np.nan_to_num(np.abs(rk4_error_hist), nan=np.inf)
        val_text = (f'  max|err| = {abs_error.max():.2e} m\n'
                    f'  MAE      = {abs_error.mean():.2e} m\n')

    pm, pc, pk = p['m'], p['c'], p['k']

    #left column text
    if p['pid_on']:
        force_line = f'  F = {p["kp"]:.1f}*e + {p["ki"]:.1f}*int(e) - {p["kd"]:.1f}*v'
        A2 = closed_loop_matrix(p)
        ss_text = (
            f'State-space, closed loop (Ki ignored):\n'
            f'  A = [[0, 1], [{A2[1,0]:.2f}, {A2[1,1]:.2f}]]\n'
            f'  B(F_ext) = [[0], [{1 / pm:.2f}]]\n'
        )
        eig_header = 'Eigenvalues (2nd-order approx):'
    else:
        force_line = '  F = external force only (PID off)'
        A = np.array([[0, 1], [-pk / pm, -pc / pm]])
        ss_text = (
            f'State-space (x_dot = A x + B F):\n'
            f'  A = [[0, 1], [{A[1,0]:.2f}, {A[1,1]:.2f}]]\n'
            f'  B = [[0], [{1 / pm:.2f}]]\n'
        )
        eig_header = 'Eigenvalues:'

    info_a.set_text(
        f'Newton\'s 2nd law:\n'
        f'  {pm:.2f}*x\'\' + {pc:.2f}*x\' + {pk:.2f}*x = F(t)\n'
        f'{force_line}\n'
        f'Effective constants (Ki root ignored):\n'
        f'  c_eff=c+Kd={c_eff:.2f}  k_eff=k+Kp={k_eff:.2f}\n'
        f'  {pm:.2f}*s^2 + {c_eff:.2f}*s + {k_eff:.2f} = 0\n'
        f'{ss_text}'
        f'{eig_header}\n'
        f'  zeta = {zeta:.3f}   wn = {wn:.3f} rad/s\n'
        f'  lambda1 = {fmt(l1)}\n'
        f'  lambda2 = {fmt(l2)}\n'
        f'  {roots}'
    )

    #right column text
    def tgt(t, keys):
        out = []
        for key in keys:
            v = t[key]
            out.append(f'{key}={v:.2f}')
        return ' '.join(out)

    snap_text = (
        f'Snap targets (one at a time):\n'
        f'  Crit:    {tgt(crit_t, ["m", "k", "c"])}\n'
        f'           {tgt(crit_t, ["kp", "kd"])}\n'
        f'  z={z_fast:.3f}: {tgt(fast_t, ["m", "k", "c"])}\n'
        f'           {tgt(fast_t, ["kp", "kd"])}\n'
        f'  z from {"SCAN" if used_scan else "FORMULA"} (v0 != 0 or Ki > 0 -> scan)\n'
        f'  Red = out of slider range\n'
        f'  Khaki = z from scan (approximate)\n'
        f'  (Kp/Kd targets assume PID on)\n'
    )

    if p['pid_on']:
        (e1, r1), (e2, r2) = root_error(pm, c_eff, k_eff, p['ki'], [l1, l2])
        ki_err_text = (f'Ki root-ignoring error vs cubic:\n'
                       f'  lambda1: {e1:.3e} ({r1:.1f}%)\n'
                       f'  lambda2: {e2:.3e} ({r2:.1f}%)\n')
        val_header = 'Validation: n/a (PID on, no exact ref)'
    else:
        ki_err_text = 'Ki root-ignoring error: n/a (PID off)\n'
        val_header = 'Validation: RK4 vs analytic (PID off)'

    info_b.set_text(
        f'{snap_text}'
        f'{ki_err_text}'
        f'{val_header}\n'
        f'{val_text}'
    )

    update_snap_colors(crit_snap_buttons, crit_t)
    update_snap_colors(fast_snap_buttons, fast_t, b_scan if used_scan else b_nthn)

    fig.canvas.draw_idle()


for slider_name in sliders:
    #update values if sliders move
    sliders[slider_name].on_changed(redraw)


def on_check(check_label):
    #check if pid is on
    if check_label == 'enable PID':
        p['pid_on'] = not p['pid_on']
    redraw()


def set_t_end(new_t_end):
    #changes t end if button is pressed, and updates graphs axis
    global t_end
    t_end = new_t_end
    btn_tend.label.set_text(f't_end: {t_end}')
    ax_ts.set_xlim(0, t_end)
    ax_rk4_error.set_xlim(0, t_end)
    ax_zeta_error.set_xlim(0, t_end)


def on_tend(_event):
    #updates everything with new t end
    set_t_end(T_END_SHORT if t_end == T_END_LONG else T_END_LONG)
    redraw()


def on_reset(_event):
    #updates anything that needs to be
    for s in sliders.values():
        s.eventson = False
        s.reset()
        s.eventson = True
    if p['pid_on']:
        chk.eventson = False
        chk.set_active(0)
        chk.eventson = True
        p['pid_on'] = False
    set_t_end(T_END_LONG)
    redraw()


btn_reset.on_clicked(on_reset)
btn_tend.on_clicked(on_tend)

chk.on_clicked(on_check)

redraw()
plt.show()


