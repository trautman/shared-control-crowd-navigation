import matplotlib.pyplot as plt
import numpy as np
from dwa_controller import DWAController
from brne_controller import BRNEController
from matplotlib.collections import LineCollection


brne_line_collection = None
dwa_line_collection = None

def visualize_brne(robot_state, goal, ped_list, ctl, ax=None, color='black'):
    global brne_line_collection

    samples = ctl.last_robot_samples
    if samples is None:
        return

    if ax is None:
        fig, ax = plt.subplots()

    # Remove previous collection
    if brne_line_collection is not None:
        brne_line_collection.remove()
        brne_line_collection = None

    # Build list of lines: each is an array of shape (tsteps, 2)
    lines = [traj for traj in samples]  # shape (N, tsteps, 2)
    brne_line_collection = LineCollection(lines, colors='black', linewidths=1, alpha=0.4)
    ax.add_collection(brne_line_collection)

    # Draw goal (optional)
    ax.plot(goal[0], goal[1], 'gx', markersize=6, zorder=10)


def visualize_dwa(robot_state, goal, obs, ctl, ax=None, color='black'):
    global dwa_line_collection

    if not hasattr(ctl, 'generate_trajectories'):
        print("[visualize_dwa] DWAController has no trajectory generator.")
        return

    samples = ctl.generate_trajectories(robot_state, goal, obs)
    if samples is None:
        return

    if ax is None:
        fig, ax = plt.subplots()

    # Remove previous collection
    if dwa_line_collection is not None:
        dwa_line_collection.remove()
        dwa_line_collection = None

    # Each traj is a list of (x, y)
    lines = [traj for traj in samples]  # list of (tsteps, 2)
    dwa_line_collection = LineCollection(lines, colors='black', linewidths=1, alpha=0.4)
    ax.add_collection(dwa_line_collection)

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



