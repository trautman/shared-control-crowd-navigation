#!/usr/bin/env python3
import argparse, os, yaml, math, random
import numpy as np
import matplotlib.pyplot as plt
import rvo2
import time

from matplotlib.patches import Rectangle, Wedge
import matplotlib.patches as mpatches
from matplotlib import patches

from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

from dwa_controller import DWAController
from brne_controller import BRNEController
from spawner_scheduler import Spawner
from visualization import visualize_dwa, visualize_brne

TIME_STEP = 0.1

def load_config(path):
    with open(path,'r') as f:
        return yaml.safe_load(f)

def add_walls(sim, walls):
    for w in walls:
        x,y,sx,sy = w['pos_x'], w['pos_y'], w['scale_x'], w['scale_y']
        x0,x1 = x - sx/2, x + sx/2
        y0,y1 = y - sy/2, y + sy/2
        corners = [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
        for i in range(4):
            sim.addObstacle([corners[i], corners[(i+1)%4]])
    sim.processObstacles()

def random_spawners_from_env(env_cfg, sim_cfg, robot_start=None):
    """
    Overrides the old uniform‐corridor sampling.
    Now we have 4 fixed “blocks”:
      block1: 3<x<7,  -11<y<-8  ↔ goal in block3 (2<x<5, 11<y<14)
      block2:10<x<11,  7<y<9   ↔ goal in block4 (3<x<7,  0<y<5)
      block3: 2<x<5,  11<y<14  ↔ goal in block1 (3<x<7, -11<y<-8)
      block4: 2.5<x<3,   -2.5<y<2.5   ↔ goal in block2 (10<x<11,7<y<9)
    Ignores robot_start since these blocks lie well outside it.
    """

    # # define your four blocks
    # blocks = {
    #   'block1': ((5,  8),   (-11, -8)),
    #   'block2': ((10, 11),   (7,   9)),
    #   'block3': ((4,  5.5),    (11,  14)),
    #   'block4': ((2.5,  3),    (-5,    2.5)),
    # }
    # # mapping start→goal
    # pair = {
    #   'block1':'block3',
    #   'block3':'block1',
    #   'block2':'block4',
    #   'block4':'block2'
    # }
    # pick blocks & pair definitions based on env file
    env_file = env_cfg.get('_env_file', '')
    if 'perks' in env_file:
        # ── PERKS environment spawner blocks ───────────────────────
        blocks = {
          'pblock1': ((0, 5), (-12, -7)),
          'pblock2': ((0, 5), (8, 10)),
          # … fill in your perks‐specific coordinate ranges …
        }
        pair = {
          'pblock1': 'pblock2',
          'pblock2': 'pblock1',
          # … etc. …
        }
    else:
        # ── BOARDWALK environment spawner blocks ────────────────
        blocks = {
          'block1': ((5,   8),    (-11,  -8)),
          'block2': ((10, 11),    (7,     9)),
          'block3': ((4,   5.5),  (11,   14)),
          'block4': ((2.5, 3),    (-5,   2.5)),
        }
        pair = {
          'block1': 'block3',
          'block3': 'block1',
          'block2': 'block4',
          'block4': 'block2',
        }

    rate_min = sim_cfg.get('spawn_rate_min', 0.1)
    rate_max = sim_cfg.get('spawn_rate_max', 1.0)
    duration = sim_cfg['simulation']['duration']

    spawners = []
    for _ in sim_cfg['ped_spawners']:
        # pick a random start block
        start_blk = random.choice(list(pair.keys()))
        x_rng, y_rng = blocks[start_blk]
        spawn = np.array([
            random.uniform(*x_rng),
            random.uniform(*y_rng)
        ])

        # pick the corresponding goal block
        goal_blk = pair[start_blk]
        gx_rng, gy_rng = blocks[goal_blk]
        goal = np.array([
            random.uniform(*gx_rng),
            random.uniform(*gy_rng)
        ])

        # spawn frequency schedule
        freq  = random.uniform(rate_min, rate_max)
        sched = [{'end': duration, 'period': freq}]

        spawners.append(Spawner(spawn, goal, sched))

    return spawners

def run_sim(env_conf, sim_conf, gui=False, output_base=None):
    print(f"GUI = {'ON' if gui else 'OFF'}")
    env_cfg = load_config(env_conf)
    sim_cfg = load_config(sim_conf)
    vis_dwa = sim_cfg.get('visualization', {}).get('dwa', False)
    vis_brne = sim_cfg.get('visualization', {}).get('brne', False)

    env_cfg['_env_file'] = os.path.basename(env_conf).lower()


    # — if loading the "perks" environment, rotate it 90° clockwise —
    if os.path.basename(env_conf).lower() == 'perks.yaml':
        def _rotate90cw(item):
            # rotate point (x,y) → (y, -x)
            x, y = item['pos_x'], item['pos_y']
            item['pos_x'], item['pos_y'] = y, -x
            # and spin the object itself by +90°
            if 'rot_z' in item:
                item['rot_z'] = (item.get('rot_z', 0.0) + 90.0) % 360.0
            return item

        # apply to floors and walls lists in place
        env_cfg['floors'] = [_rotate90cw(f) for f in env_cfg.get('floors', [])]
        env_cfg['walls']  = [_rotate90cw(w) for w in env_cfg.get('walls', [])]

    # ── Visualization settings ───────────────────────────────
    viz_cfg      = sim_cfg.get('visualization', {})
    report_interval = float(
        sim_cfg.get('simulation', {}).get('report_interval', 10.0))
    ENABLE_DWA_VIZ  = viz_cfg.get('dwa', False)
    ENABLE_BRNE_VIZ = viz_cfg.get('brne', False)
    VIZ_PAUSE     = viz_cfg.get('pause_between', TIME_STEP)


    floors = env_cfg.get('floors',[])
    walls  = env_cfg.get('walls', [])

    S  = sim_cfg['simulation']
    duration    = S['duration']
    robot_delay = S['robot_delay']
    goal_tol    = S['goal_tolerance']
    close_th    = S['close_stop_threshold']
    agent_speed = S.get('agent_speed',1.0)
    n_trials    = S.get('n_trials',1)

    robot_cfgs = sim_cfg['robots']  # list of dicts

    base = output_base or '.'

    for trial in range(1, n_trials+1):
        print(f"\n=== Trial {trial}/{n_trials} ===")
        # record real‐wall clock start for this trial
        trial_wall_start = time.perf_counter()

        next_report = report_interval

        # spawners
        if sim_cfg.get('spawn_mode','fixed').lower()=='random':
            ped_spawners = random_spawners_from_env(env_cfg, sim_cfg, None)
        else:
            ped_spawners=[]
            for sp in sim_cfg['ped_spawners']:
                pos = np.array(sp['pos'])
                goal_pt = np.array(sp['goal'])
                freq = sp.get('freq', sp.get('spawn_frequency',1.0))
                sched= [{'end':duration,'period':freq}]
                ped_spawners.append(Spawner(pos,goal_pt,sched))

        ped_agents=[]

        # ORCA sim
        O = sim_cfg['orca']
        sim = rvo2.PyRVOSimulator(
            TIME_STEP,
            O['neighbor_dist'],O['max_neighbors'],
            O['time_horizon'],O['time_horizon_obst'],
            O['agent_radius'], agent_speed
        )

        robots = []
        robot_colors = ['magenta', 'cyan', 'blue', 'orange', 'green']
        for i, R in enumerate(sim_cfg['robots']):
            color = robot_colors[i % len(robot_colors)]  # Get a distinct color for each robot
            algo = R['algorithm'].upper()
            start = np.array(R['start'])
            goal = np.array(R['goal'])
            th0 = math.atan2(goal[1] - start[1], goal[0] - start[0])
            rstate = [start[0], start[1], th0, 0.0, 0.0]

            fov_deg = R.get('fov', {}).get('degree', 104.0)  # default fallback
            fov_range = R.get('fov', {}).get('range', 2.0)

            if algo == 'BRNE':
                rcfg = sim_cfg['brne']
                ctl = BRNEController(rcfg, TIME_STEP)
            else:
                rcfg = sim_cfg['dwa']
                ctl = DWAController(rcfg, TIME_STEP)

            rid = sim.addAgent(tuple(start))
            sim.setAgentMaxSpeed(rid, rcfg['max_speed'])
            robots.append({
                'id': i,
                'algo': algo,
                'start': start,
                'goal': goal,
                'ctl': ctl,
                'rstate': rstate,
                'rid': rid,
                'color': color,
                'fov_deg': fov_deg,
                'fov_range': fov_range,
                'metrics': {
                    'density': [],
                    'safety': [],
                    'trans_vel': [],
                    'path_len': [],
                    'elapsed': [],
                    'stopped': [],
                    'prev_pos': start.copy(),
                }
            })

        # --- DEBUG: print each robot's algorithm ---
        for robot in robots:
            print(f"Robot {robot['id']+1} algorithm = {robot['algo']}")

        add_walls(sim, walls)
        
        # metrics storage
        density              = []
        safety_distances     = []
        translational_vels   = []
        dist_traveled        = []
        time_elapsed_list    = []
        time_not_moving_list = []
        prev_pos = sim.getAgentPosition(rid)

        # fov_area = (FOV_DEG/360.0)*math.pi*(FOV_R**2)
        t=0.0
        achieved=False

        # initial robot state
        # th0 = math.atan2(goal[1]-start[1], goal[0]-start[0])
        rstate = [start[0], start[1], th0, 0.0, 0.0]

        # GUI setup
        if gui:
            plt.ion()
            mag = sim_cfg.get("simulation", {}).get("sim_size_magnification", 1.0)
            fig, ax = plt.subplots(figsize=(8 * mag, 8 * mag))

            # fig,ax = plt.subplots(figsize=(12,12))
            ax.set_aspect('equal')

            # floors & walls
            for f in floors:
                fx,fy = f['pos_x'], f['pos_y']
                fw,fh = f['scale_x'],f['scale_y']
                ang   = f.get('rot_z',0.0)
                rect  = Rectangle((-fw/2,-fh/2),fw,fh,color='lightgray',zorder=0)
                rect.set_transform(Affine2D().rotate_deg_around(0,0,-ang)
                                   .translate(fx,fy)+ax.transData)
                ax.add_patch(rect)
            for w in walls:
                ax.add_patch(Rectangle(
                    (w['pos_x']-w['scale_x']/2, w['pos_y']-w['scale_y']/2),
                    w['scale_x'], w['scale_y'], color='saddlebrown', zorder=1
                ))
            # spawners & goal/start
            for sp in ped_spawners:
                ax.scatter(*sp.pos, c='green', marker='x', s=80, zorder=2)
                ax.scatter(*sp.goal,c='red',   marker='*', s=80, zorder=2)

            # plot each robot’s start and goal in its own color,
            # and label the legend “Robot <i>: <ALGO>”
            for idx, robot in enumerate(robots):
                color = robot['color']
                algo  = robot['algo']  # should be 'DWA' or 'BRNE', etc.

                # square = start; diamond = goal
                ax.scatter(
                    *robot['start'],
                    c=color, marker='s', s=80,
                    zorder=3,
                    label=f"Robot {idx+1}: {algo}"
                )
                ax.scatter(
                    *robot['goal'],
                    c=color, marker='D', s=80,
                    zorder=3
                )

            # now draw a single legend
            ax.legend(loc='upper left')

            scatter = ax.scatter([],[],s=50,zorder=4)

            robot_patches = []
            robot_wedges = []

            for robot in robots:
                color = robot['color']

                patch = Rectangle((-0.3, -0.3), 0.6, 0.6,
                                  facecolor=color, edgecolor='black', zorder=5)
                ax.add_patch(patch)
                robot_patches.append(patch)

                fov_range = R.get('fov', {}).get('range', 2.0)
                wedge = patches.Wedge((0, 0), fov_range, 0, 0, color=color, alpha=0.15)

                ax.add_patch(wedge)
                robot_wedges.append(wedge)

            # ── NEW: Create a Text label for each robot’s algorithm ───────────
            robot_texts = []
            for idx, robot in enumerate(robots):
                x0, y0 = robot['start'][:2]
                txt = ax.text(
                    x0, y0 + 0.4,               # 0.4m above the robot
                    robot['algo'],              # label = "DWA" or "BRNE"
                    color=robot['color'],       # same color as robot
                    fontsize=9,
                    ha='center',
                    va='bottom',
                    zorder=6
                )
                robot_texts.append(txt)
            # ─────────────────────────────────────────

            # Build one legend handle per robot:
            legend_handles = []
            for idx, robot in enumerate(robots):
                color = robot['color']
                algo  = robot['algo']  # 'DWA' or 'BRNE'
                legend_handles.append(
                    Line2D([0], [0],
                        marker='s',         # same marker shape you used
                        color=color,
                        label=f"Robot {idx+1}: {algo}",
                        markersize=8,
                        linestyle=''        # no line connecting the markers
                    )
                )

            # Now place the legend:
            ax.legend(handles=legend_handles, loc='upper left')

            plt.draw(); plt.pause(0.001)
            frame_count = 0
 
            ne_collection = LineCollection(
                [], linewidths=2, colors='black', zorder=3
            )
            ax.add_collection(ne_collection)
            
            # ── Pre-allocate DWA visualization ────────────────────────────────
            dwa_cands = LineCollection(
                [], linewidths=0.5, alpha=0.3, colors='blue', zorder=2
            )
            ax.add_collection(dwa_cands)
            
            dwa_best_line, = ax.plot(
                [], [], linewidth=2, color='darkblue', label='DWA best', zorder=3
            )
            ax.legend(loc='upper left')

            gp_lines = []
            ne_lines = []
            frame_count = 0

        frame_count = 0
        # run simulation loop
        while t < duration:
            frame_count += 1
            dt = TIME_STEP
            
            for sp in ped_spawners:
                p = sp.current_period(t)
                if t >= sp.last_spawn + p:
                    aid = sim.addAgent(tuple(sp.pos))
                    ped_agents.append({'id':aid,'goal':sp.goal})
                    sp.last_spawn = t

            fov_deg = robot['fov_deg']
            fov_range = robot['fov_range']
            vis = []
            rx, ry, th = rstate[:3]
            for a in ped_agents:
                px, py = sim.getAgentPosition(a['id'])
                dx, dy = px - rx, py - ry
                d = math.hypot(dx, dy)
                ang = math.degrees(math.atan2(dy, dx) - th)
                rel = (ang + 180) % 360 - 180
                if d <= fov_range and abs(rel) <= fov_deg / 2:
                    vis.append({'dist': d, 'goal': a['goal']})

            # robot control
            prev = list(rstate)
            if t>=robot_delay:
                for robot in robots:
                    ctl = robot['ctl']
                    algo = robot['algo']
                    goal = robot['goal']
                    rstate = robot['rstate']
                    fov_range = robot['fov_range']
                    fov_deg = robot['fov_deg']
                    rid = robot['rid']
                    metrics = robot['metrics']

                    rx, ry, th = rstate[:3]
                    in_fov = []
                    for a in ped_agents:
                        px, py = sim.getAgentPosition(a['id'])
                        dx, dy = px - rx, py - ry
                        d = math.hypot(dx, dy)
                        ang = math.degrees(math.atan2(dy, dx) - th)
                        rel = (ang + 180) % 360 - 180
                        if d <= fov_range and abs(rel) <= fov_deg / 2:
                            in_fov.append({
                                'id': a['id'],
                                'pos': (px, py),
                                'goal': a['goal'],
                                'dist': d
                            })
                    viz_cfg = sim_cfg.get("visualization", {})
                    brne_viz_cfg = viz_cfg.get("brne", {}) if isinstance(viz_cfg.get("brne", {}), dict) else {}
                    dwa_viz_cfg  = viz_cfg.get("dwa", {}) if isinstance(viz_cfg.get("dwa", {}), dict) else {}
                    if algo == 'DWA':
                        obs = [p['pos'] for p in in_fov]
                        ctl.cfg['current_v'] = rstate[3]
                        v, w = ctl.control(rstate, goal, obs)
                        if vis_dwa and 'ax' in locals():
                            visualize_dwa(rstate, goal, obs, ctl, ax=ax, color=robot['color'], cfg=dwa_viz_cfg)
                    else:
                        ped_list_fov = [{'id': p['id'], 'pos': p['pos'], 'goal': p['goal']} for p in in_fov]
                        v, w = ctl.control(rstate, goal, ped_list_fov)
                        if vis_brne and 'ax' in locals():
                            visualize_brne(rstate, goal, ped_list_fov, ctl, ax=ax, color=robot['color'], cfg=brne_viz_cfg)

                    # ✅ Clamp and record
                    v = np.clip(v, -ctl.cfg['max_speed'], ctl.cfg['max_speed'])
                    w = np.clip(w, -ctl.cfg['max_yaw_rate'], ctl.cfg['max_yaw_rate'])
                    robot['last_cmd'] = (v, w)

                    rx, ry = robot['rstate'][0], robot['rstate'][1]
                    distances = []
                    for a in ped_agents:
                        px, py = sim.getAgentPosition(a['id'])
                        dist = math.hypot(px - rx, py - ry)
                        distances.append(dist)

                    if distances:
                        min_dist = min(distances)
                    else:
                        min_dist = 10.0  # default safety distance if no pedestrians present

                    robot['metrics']['safety'].append(min_dist)

                    if not np.isfinite(min_dist):
                        min_dist = 5.0
                    robot['metrics']['safety'].append(min_dist)

                    # ─── Density: number of agents in FOV per m² ───
                    fov_area = (robot['fov_deg'] / 360.0) * math.pi * (robot['fov_range'] ** 2)
                    agents_in_fov = 0
                    for a in ped_agents:
                        px, py = sim.getAgentPosition(a['id'])
                        dx, dy = px - rx, py - ry
                        d = math.hypot(dx, dy)
                        angle = math.degrees(math.atan2(dy, dx) - robot['rstate'][2])
                        angle = (angle + 180) % 360 - 180
                        if d <= robot['fov_range'] and abs(angle) <= robot['fov_deg'] / 2:
                            agents_in_fov += 1
                    density_val = agents_in_fov / fov_area if fov_area > 0 else 0.0
                    metrics['density'].append(density_val)

                    robot_th = robot['rstate'][2]
                    sim.setAgentPrefVelocity(
                        rid,
                        (v * math.cos(robot_th),
                        v * math.sin(robot_th))
                    )

                    sim.setAgentPosition(rid, (rstate[0], rstate[1]))

                    cur_pos = (robot['rstate'][0], robot['rstate'][1])
                    dx, dy = cur_pos[0] - metrics['prev_pos'][0], cur_pos[1] - metrics['prev_pos'][1]
                    dist = math.hypot(dx, dy)                    
                    metrics['path_len'].append(dist)
                    metrics['trans_vel'].append(v)
                    metrics['elapsed'].append(TIME_STEP)
                    metrics['stopped'].append(0.0 if abs(v) > 1e-3 or abs(w) > 1e-3 else TIME_STEP)
                    metrics['prev_pos'] = cur_pos
            else:
                v,w = 0.0, 0.0
                sim.setAgentPrefVelocity(rid,(0.0,0.0))

            # ped pref-vel
            for a in ped_agents:
                p = np.array(sim.getAgentPosition(a['id']))
                vec = a['goal'] - p; dd = np.linalg.norm(vec)
                pv = tuple((vec/dd)*agent_speed if dd>1e-3 else (0.0,0.0))
                sim.setAgentPrefVelocity(a['id'],pv)

            sim.doStep()

            # now update each robot’s state and ORCA agent
            for robot in robots:
                rid    = robot['rid']
                rstate = robot['rstate']
                v, w   = robot['last_cmd']

                # integrate motion
                th = rstate[2]
                rstate[0] += v * math.cos(th) * TIME_STEP
                rstate[1] += v * math.sin(th) * TIME_STEP
                rstate[2] = (rstate[2] + w * TIME_STEP + math.pi) % (2*math.pi) - math.pi
                rstate[3], rstate[4] = v, w

                # push to ORCA sim
                sim.setAgentPosition(rid, (rstate[0], rstate[1]))
                sim.setAgentVelocity(rid, (v*math.cos(rstate[2]), v*math.sin(rstate[2])))

            # GUI update
            if gui:
                # build positions + per-agent colors based on ANY robot’s FOV
                pts    = [sim.getAgentPosition(a['id']) for a in ped_agents]
                colors = []
                for (x, y), a in zip(pts, ped_agents):
                    in_any_fov = False
                    for robot in robots:
                        fov_range = robot['fov_range']
                        fov_deg = robot['fov_deg']
                        rx, ry, rth = robot['rstate'][:3]
                        dx, dy  = x - rx, y - ry
                        dist    = math.hypot(dx, dy)
                        rel_ang = (math.degrees(math.atan2(dy, dx) - rth) + 180) % 360 - 180
                        if dist <= fov_range and abs(rel_ang) <= fov_deg / 2:
                            in_any_fov = True
                            break
                    colors.append('red' if in_any_fov else 'gray')
                if pts:
                    scatter.set_offsets(pts)
                    scatter.set_color(colors)

                for robot, patch, wedge, text in zip(robots, robot_patches, robot_wedges, robot_texts):
                    x, y, th = robot['rstate'][:3]

                    fov_deg = robot['fov_deg']         # ✅ per-robot FOV
                    fov_range = robot['fov_range']     # ✅ per-robot range

                    tr = Affine2D().rotate_deg_around(0, 0, math.degrees(th)).translate(x, y)
                    patch.set_transform(tr + ax.transData)

                    wedge.set_center((x, y))
                    wedge.set_radius(fov_range)        # ✅ optional: in case radius needs updating
                    th_deg = math.degrees(th)
                    wedge.theta1 = th_deg - fov_deg / 2
                    wedge.theta2 = th_deg + fov_deg / 2

                    text.set_position((x, y + 0.4))
                plt.pause(TIME_STEP)

            # prune
            ped_agents = [a for a in ped_agents
                if np.linalg.norm(np.array(sim.getAgentPosition(a['id']))-a['goal'])>0.5]

            # check goal
            if not achieved and t>=TIME_STEP and np.linalg.norm(rstate[:2]-goal)<goal_tol:
                achieved = True
                break

            t += TIME_STEP

            # ── report progress every report_interval s ──────────────────────────────
            if t >= next_report:
                # compute wall‐clock elapsed
                wall_elapsed = time.perf_counter() - trial_wall_start
                print(
                    f"[Trial {trial}] sim {t:.1f}/{duration:.1f}s    "
                    f"real {wall_elapsed:.1f}s"
                )
                next_report += report_interval

        for robot in robots:
            i = robot['id'] + 1  # robot1 = 1
            metrics = robot['metrics']
            robot_dir = os.path.join(base, f"robot_{i}")
            os.makedirs(robot_dir, exist_ok=True)

            def save_metric(name, data):
                subdir = os.path.join(robot_dir, name)
                os.makedirs(subdir, exist_ok=True)
                out = os.path.join(subdir, f"{name}_trial_{trial}.txt")
                with open(out, 'w') as f:
                    for val in data:
                        f.write(f"{val:.6f}\n")

            save_metric("density", metrics['density'])
            save_metric("safety_distances", metrics['safety'])
            save_metric("translational_velocity", metrics['trans_vel'])
            save_metric("path_length", metrics['path_len'])
            save_metric("travel_time", metrics['elapsed'])
            save_metric("time_not_moving", metrics['stopped'])
        if gui:
            plt.clf()

    print("\nAll done.")

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env-config', default='boardwalk.yaml')
    p.add_argument('--sim-config', default='simulator_config.yaml')
    p.add_argument('-g','--gui', action='store_true',
                   help='Live-plot each trial')
    p.add_argument('--write_data_to', default=None,
                   help='Optional base directory for saving metric outputs')
    args = p.parse_args()
    run_sim(args.env_config, args.sim_config, gui=args.gui, output_base=args.write_data_to)

