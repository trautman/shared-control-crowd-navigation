import matplotlib.pyplot as plt
import numpy as np
from dwa_controller import DWAController
from brne_controller import BRNEController
from matplotlib.collections import LineCollection


brne_line_collection = None
dwa_line_collection = None

# Add globals for optimal trajectory lines
brne_optimal_line = None
dwa_optimal_line = None



def visualize_brne(robot_state, goal, ped_list, ctl, ax=None, color='black', cfg=None):
    global brne_line_collection, brne_optimal_line

    samples = ctl.last_robot_samples  # shape (N, tsteps, 2)
    weights = ctl.last_W              # shape (1, N)
    if samples is None or weights is None:
        return

    if ax is None:
        fig, ax = plt.subplots()

    if cfg is None:
        cfg = {'all_samples': True, 'optima': True}

    if cfg.get('all_samples', True):
        lines = [traj for traj in samples]
        if brne_line_collection is None:
            brne_line_collection = LineCollection(lines, colors='black', linewidths=1, alpha=0.3)
            ax.add_collection(brne_line_collection)
        else:
            brne_line_collection.set_segments(lines)

    if brne_optimal_line is not None:
        brne_optimal_line.remove()
        brne_optimal_line = None

    if cfg.get('optima', True):
        w0 = weights[0]
        if w0.sum() > 0:
            i_star = np.argmax(w0)
            traj_star = samples[i_star]
            brne_optimal_line, = ax.plot(traj_star[:, 0], traj_star[:, 1],
                                         color='black', linewidth=2.5, zorder=10)

    ax.plot(goal[0], goal[1], 'gx', markersize=6, zorder=10)




def visualize_dwa(robot_state, goal, obs, ctl, ax=None, color='black', cfg=None):
    global dwa_line_collection, dwa_optimal_line

    if not hasattr(ctl, 'generate_trajectories'):
        print("[visualize_dwa] DWAController has no trajectory generator.")
        return

    samples = ctl.generate_trajectories(robot_state, goal, obs)
    if samples is None:
        return

    if ax is None:
        fig, ax = plt.subplots()

    if cfg is None:
        cfg = {'all_samples': True, 'optima': True}

    if cfg.get('all_samples', True):
        lines = [traj for traj in samples]
        if dwa_line_collection is None:
            dwa_line_collection = LineCollection(lines, colors='black', linewidths=1, alpha=0.3)
            ax.add_collection(dwa_line_collection)
        else:
            dwa_line_collection.set_segments(lines)

    if dwa_optimal_line is not None:
        dwa_optimal_line.remove()
        dwa_optimal_line = None

    if cfg.get('optima', True):
        best = ctl.control(robot_state, goal, obs)
        best_traj = ctl.predict_trajectory(robot_state, *best)
        xs, ys = zip(*best_traj)
        dwa_optimal_line, = ax.plot(xs, ys, color='black', linewidth=2.5, zorder=10)

    ax.plot(goal[0], goal[1], 'gx', markersize=6, zorder=10)



def draw_gp_samples(ax, state, obstacles, controller, num_samples=20, horizon=5):
    """
    Draw num_samples GP trajectories of length “horizon” on the given Axes ax.
    Assumes controller has a method `sample_gp(state, obstacles, num_samples, horizon)`
    that returns an array of shape (num_samples, horizon, 2).
    """
    # get your GP trajectories
    samples = controller.sample_gp(state, obstacles,
                                   num_samples=num_samples,
                                   horizon=horizon)
    # plot each sample
    for traj in samples:
        ax.plot(traj[:, 0], traj[:, 1],
                linewidth=1,
                alpha=0.3)



