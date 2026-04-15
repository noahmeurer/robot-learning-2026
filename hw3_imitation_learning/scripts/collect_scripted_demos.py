"""Scripted (oracle) multicube data collector for goal-conditioned IL.

Generates expert-quality demonstrations by executing a deterministic waypoint
planner that uses privileged cube/bin positions from the simulation.  Writes
raw zarr data in the exact same schema as ``record_teleop_demos.py --multicube``
so that ``compute_actions.py`` and ``train.py`` work without modification.

Supports visual validation (default) and headless fast collection.

Usage:
    python scripts/collect_scripted_demos.py --num-episodes 100 --headless
    python scripts/collect_scripted_demos.py --num-episodes 10 --goal-cube red
    python scripts/collect_scripted_demos.py --num-episodes 300 --seed 42 --headless
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from hw3.eval_utils import (
    check_cube_out_of_bounds,
    check_success,
    check_wrong_cube_in_bin,
)
from hw3.sim_env import (
    CUBE_COLORS,
    CUBE_JOINT_NAMES,
    GOAL_DIM,
    SO100MulticubeSimEnv,
)
from hw3.teleop_utils import (
    CAMERA_NAMES,
    JOINT_NAMES,
    compose_camera_views,
)
from so101_gym.constants import ASSETS_DIR

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from record_teleop_demos import MulticubeZarrWriter

# ── planner constants (metres) ────────────────────────────────────────
SAFE_Z = 0.2
GRASP_Z = 0.065
RELEASE_Z = 0.2
MOVE_STEP = 0.01
XY_TOL = 0.012
Z_TOL = 0.008
SETTLE_STEPS = 5
GRIPPER_OPEN_VAL = 0.8
GRIPPER_CLOSE_VAL = -0.15

# Grasp XY offset: added to the target cube position so the cube lands
# between the gripper jaws rather than centred on the EE site.  Tune these
# if the fixed jaw doesn't contact the cube properly.
#   +x = shift EE to the right of the cube
#   +y = shift EE further from the robot base
GRASP_OFFSET_X = -0.013
GRASP_OFFSET_Y = 0.0

# Release XY offset: same idea but for the bin drop position.
RELEASE_OFFSET_X = -0.013

XML_PATH = ASSETS_DIR / "so100_multicube_ee.xml"


class Phase(Enum):
    RAISE_SAFE = auto()
    XY_TO_CUBE = auto()
    OPEN_DESCEND = auto()
    CLOSE_SETTLE = auto()
    LIFT_SAFE = auto()
    XY_TO_BIN = auto()
    DESCEND_RELEASE = auto()
    RETREAT = auto()
    POST_SUCCESS = auto()


# ── helpers ───────────────────────────────────────────────────────────


def move_toward(current: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    diff = target - current
    dist = float(np.linalg.norm(diff))
    if dist <= max_step:
        return target.copy()
    return current + (diff / dist) * max_step


def at_target(current: np.ndarray, target: np.ndarray, tol: float) -> bool:
    return bool(np.linalg.norm(current - target) < tol)


def record_step(
    env: SO100MulticubeSimEnv,
    writer: MulticubeZarrWriter,
    goal_onehot: np.ndarray,
    goal_pos: np.ndarray,
) -> None:
    state_joints = env.get_joint_angles()
    state_ee = env.get_ee_state()
    state_cubes = env.get_all_cubes_state()
    state_gripper = np.array([env.get_gripper_angle()], dtype=np.float32)
    action_gripper = np.array(
        [env.data.ctrl[env.act_ids[env._jaw_idx]]], dtype=np.float32
    )
    dummy_obstacle = np.zeros(3, dtype=np.float32)
    writer.append_with_goal(
        state_joints,
        state_ee,
        state_cubes,
        state_gripper,
        action_gripper,
        dummy_obstacle,
        goal_onehot,
        goal_pos,
    )


def compose_views(env: SO100MulticubeSimEnv) -> np.ndarray:
    images = {cam: env.render(cam) for cam in CAMERA_NAMES}
    return compose_camera_views(images, CAMERA_NAMES)


# ── goal schedule ─────────────────────────────────────────────────────


def build_goal_schedule(
    goal_cube: str,
    num_episodes: int,
    sampling: str,
    rng: np.random.Generator,
) -> list[str]:
    if goal_cube != "all":
        return [goal_cube] * num_episodes
    if sampling == "cycle":
        return [CUBE_COLORS[i % len(CUBE_COLORS)] for i in range(num_episodes)]
    # uniform random
    return [CUBE_COLORS[int(rng.integers(len(CUBE_COLORS)))] for _ in range(num_episodes)]


# ── single episode ────────────────────────────────────────────────────


def run_scripted_episode(
    env: SO100MulticubeSimEnv,
    writer: MulticubeZarrWriter,
    goal_color: str,
    *,
    max_steps: int,
    post_success_steps: int,
    headless: bool,
    ep_idx: int,
    successes_so_far: int,
    total_so_far: int,
) -> tuple[bool, str | None]:
    """Run one scripted pick-and-place episode.

    Returns ``(success, failure_reason)``.
    ``failure_reason`` is ``None`` on success, otherwise one of
    ``"timeout"``, ``"out_of_bounds"``, ``"wrong_cube"``, ``"user_abort"``.
    """
    env.set_goal(goal_color)
    obs = env.reset()

    goal_onehot = env.get_goal_onehot().astype(np.float32)
    goal_pos = env.get_goal_pos().astype(np.float32)
    target_cube_xyz = env.get_target_cube_state()[:3].copy()
    bin_center_xy = goal_pos[:2].copy()

    # Grasp target: cube XY + user-configurable offset for asymmetric gripper
    grasp_xy = target_cube_xyz[:2].copy()
    grasp_xy[0] += GRASP_OFFSET_X
    grasp_xy[1] += GRASP_OFFSET_Y

    # Commanded position tracks the mocap target independently of actual EE.
    cmd_pos = env.get_ee_pos().copy()

    phase = Phase.RAISE_SAFE
    settle_counter = 0
    tail_counter = 0
    phase_steps = 0
    success_detected = False

    window_name = "Scripted Multicube Collection"

    for step in range(max_steps):
        phase_steps += 1

        # ── 1. advance commanded position and set gripper ─────────────
        if phase == Phase.RAISE_SAFE:
            goal = np.array([cmd_pos[0], cmd_pos[1], SAFE_Z])
            cmd_pos = move_toward(cmd_pos, goal, MOVE_STEP)

        elif phase == Phase.XY_TO_CUBE:
            goal = np.array([grasp_xy[0], grasp_xy[1], SAFE_Z])
            cmd_pos = move_toward(cmd_pos, goal, MOVE_STEP)

        elif phase == Phase.OPEN_DESCEND:
            live_cube = env.get_target_cube_state()[:3]
            goal = np.array([live_cube[0] + GRASP_OFFSET_X,
                             live_cube[1] + GRASP_OFFSET_Y, GRASP_Z])
            cmd_pos = move_toward(cmd_pos, goal, MOVE_STEP)
            env.set_gripper(GRIPPER_OPEN_VAL)

        elif phase == Phase.CLOSE_SETTLE:
            env.set_gripper(GRIPPER_CLOSE_VAL)

        elif phase == Phase.LIFT_SAFE:
            goal = np.array([cmd_pos[0], cmd_pos[1], SAFE_Z])
            cmd_pos = move_toward(cmd_pos, goal, MOVE_STEP)

        elif phase == Phase.XY_TO_BIN:
            goal = np.array([bin_center_xy[0] + RELEASE_OFFSET_X, bin_center_xy[1], SAFE_Z])
            cmd_pos = move_toward(cmd_pos, goal, MOVE_STEP)

        elif phase == Phase.DESCEND_RELEASE:
            goal = np.array([bin_center_xy[0] + RELEASE_OFFSET_X, bin_center_xy[1], RELEASE_Z])
            cmd_pos = move_toward(cmd_pos, goal, MOVE_STEP)

        elif phase == Phase.RETREAT:
            goal = np.array([cmd_pos[0], cmd_pos[1], SAFE_Z])
            cmd_pos = move_toward(cmd_pos, goal, MOVE_STEP)
            env.set_gripper(GRIPPER_OPEN_VAL)

        elif phase == Phase.POST_SUCCESS:
            env.set_gripper(GRIPPER_OPEN_VAL)

        env.set_mocap_pos(cmd_pos)

        # ── 2. record current state ──────────────────────────────────
        record_step(env, writer, goal_onehot, goal_pos)

        # ── 3. step physics ───────────────────────────────────────────
        obs = env.step()
        ee_pos = env.get_ee_pos()

        # ── 4. check success / failure ────────────────────────────────
        if not success_detected and check_success(env):
            success_detected = True
            phase = Phase.POST_SUCCESS
            phase_steps = 0
            tail_counter = 0

        if phase == Phase.POST_SUCCESS:
            tail_counter += 1
            if tail_counter >= post_success_steps:
                return True, None

        wrong = check_wrong_cube_in_bin(env)
        if wrong is not None:
            return False, "wrong_cube"

        if check_cube_out_of_bounds(env):
            return False, "out_of_bounds"

        # ── 5. phase transitions ──────────────────────────────────────
        # Primary: actual EE within tolerance of the phase goal.
        # Fallback: if the commanded position has reached the goal and
        # the EE has had SETTLE_STEPS extra steps to converge, transition
        # anyway (handles cases where arm kinematics prevent exact arrival).

        def _cmd_arrived(goal_pt: np.ndarray) -> bool:
            return float(np.linalg.norm(cmd_pos - goal_pt)) < 1e-5

        if phase == Phase.RAISE_SAFE:
            goal_pt = np.array([cmd_pos[0], cmd_pos[1], SAFE_Z])
            arrived = abs(ee_pos[2] - SAFE_Z) < Z_TOL
            stalled = _cmd_arrived(goal_pt) and phase_steps > SETTLE_STEPS
            if arrived or stalled:
                phase = Phase.XY_TO_CUBE
                phase_steps = 0

        elif phase == Phase.XY_TO_CUBE:
            goal_pt = np.array([grasp_xy[0], grasp_xy[1], SAFE_Z])
            arrived = np.linalg.norm(ee_pos[:2] - grasp_xy) < XY_TOL
            stalled = _cmd_arrived(goal_pt) and phase_steps > SETTLE_STEPS
            if arrived or stalled:
                phase = Phase.OPEN_DESCEND
                phase_steps = 0

        elif phase == Phase.OPEN_DESCEND:
            live_cube = env.get_target_cube_state()[:3]
            goal_pt = np.array([live_cube[0] + GRASP_OFFSET_X,
                                live_cube[1] + GRASP_OFFSET_Y, GRASP_Z])
            arrived = abs(ee_pos[2] - GRASP_Z) < Z_TOL
            stalled = _cmd_arrived(goal_pt) and phase_steps > SETTLE_STEPS
            if arrived or stalled:
                phase = Phase.CLOSE_SETTLE
                settle_counter = 0
                phase_steps = 0

        elif phase == Phase.CLOSE_SETTLE:
            settle_counter += 1
            if settle_counter >= SETTLE_STEPS:
                phase = Phase.LIFT_SAFE
                phase_steps = 0

        elif phase == Phase.LIFT_SAFE:
            goal_pt = np.array([cmd_pos[0], cmd_pos[1], SAFE_Z])
            arrived = abs(ee_pos[2] - SAFE_Z) < Z_TOL
            stalled = _cmd_arrived(goal_pt) and phase_steps > SETTLE_STEPS
            if arrived or stalled:
                phase = Phase.XY_TO_BIN
                phase_steps = 0

        elif phase == Phase.XY_TO_BIN:
            bin_target_xy = np.array([bin_center_xy[0] + RELEASE_OFFSET_X, bin_center_xy[1]])
            goal_pt = np.array([bin_target_xy[0], bin_target_xy[1], SAFE_Z])
            arrived = np.linalg.norm(ee_pos[:2] - bin_target_xy) < XY_TOL
            stalled = _cmd_arrived(goal_pt) and phase_steps > SETTLE_STEPS
            if arrived or stalled:
                phase = Phase.DESCEND_RELEASE
                phase_steps = 0

        elif phase == Phase.DESCEND_RELEASE:
            goal_pt = np.array([bin_center_xy[0] + RELEASE_OFFSET_X, bin_center_xy[1], RELEASE_Z])
            arrived = abs(ee_pos[2] - RELEASE_Z) < Z_TOL
            stalled = _cmd_arrived(goal_pt) and phase_steps > SETTLE_STEPS
            if arrived or stalled:
                env.set_gripper(GRIPPER_OPEN_VAL)
                phase = Phase.RETREAT
                phase_steps = 0

        elif phase == Phase.RETREAT:
            goal_pt = np.array([cmd_pos[0], cmd_pos[1], SAFE_Z])
            arrived = abs(ee_pos[2] - SAFE_Z) < Z_TOL
            stalled = _cmd_arrived(goal_pt) and phase_steps > SETTLE_STEPS
            if arrived or stalled:
                if success_detected:
                    phase = Phase.POST_SUCCESS
                    phase_steps = 0
                    tail_counter = 0

        # ── 6. render (visual mode) ──────────────────────────────────
        if not headless:
            img = compose_views(env)
            status = (
                f"ep {ep_idx} | goal: {goal_color} | "
                f"phase: {phase.name} | step {step}/{max_steps}"
            )
            cv2.putText(img, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            sr = successes_so_far / total_so_far * 100 if total_so_far > 0 else 0
            sr_text = f"Saved: {successes_so_far}/{total_so_far} ({sr:.0f}%)"
            cv2.putText(img, sr_text, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return False, "user_abort"
            time.sleep(env.dt_ctrl)

    return False, "timeout"


# ── main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect scripted multicube demonstrations for goal-conditioned IL."
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        required=True,
        help="Number of *successful* episodes to collect.",
    )
    parser.add_argument("--max-steps", type=int, default=800, help="Max steps per episode attempt (default: 800).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for layout and goal scheduling.")
    parser.add_argument("--headless", action="store_true", help="Disable rendering for fast collection.")
    parser.add_argument("--control-hz", type=float, default=10.0, help="Control frequency (default: 10.0).")
    parser.add_argument("--render-w", type=int, default=640, help="Render width (default: 640).")
    parser.add_argument("--render-h", type=int, default=480, help="Render height (default: 480).")
    parser.add_argument("--out", type=Path, default=None, help="Explicit output .zarr path.")
    parser.add_argument(
        "--goal-cube",
        type=str,
        default="all",
        choices=["red", "green", "blue", "all"],
        help="Goal colour or 'all' for mixed (default: all).",
    )
    parser.add_argument(
        "--goal-sampling",
        type=str,
        default="uniform",
        choices=["cycle", "uniform"],
        help="Goal colour scheduling when --goal-cube=all (default: uniform).",
    )
    parser.add_argument("--no-shuffle", action="store_true", help="Disable slot shuffling.")
    parser.add_argument("--post-success-steps", type=int, default=32, help="Tail steps to record after success (default: 32).")
    parser.add_argument("--min-steps-per-episode", type=int, default=20, help="Discard episodes shorter than this (default: 20).")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # We over-allocate the schedule since failed attempts consume entries.
    schedule_size = args.num_episodes * 4
    goal_schedule = build_goal_schedule(args.goal_cube, schedule_size, args.goal_sampling, rng)

    # ── output path ───────────────────────────────────────────────────
    ts = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H-%M-%S")
    if args.out is not None:
        out_zarr = args.out
    else:
        run_dir = Path("./datasets/raw/multi_cube/teleop") / ts
        out_zarr = run_dir / "so100_multicube_teleop.zarr"
    out_zarr.parent.mkdir(parents=True, exist_ok=True)

    # ── writer ────────────────────────────────────────────────────────
    xml_path = XML_PATH
    writer = MulticubeZarrWriter(
        out_zarr,
        joint_dim=len(JOINT_NAMES),
        ee_dim=7,
        cube_dim=0,
        gripper_dim=1,
        obstacle_dim=3,
        flush_every=12,
    )
    writer.set_attrs(
        xml=str(xml_path),
        joint_names=list(JOINT_NAMES),
        cube_colors=list(CUBE_COLORS),
        cube_joint_names=list(CUBE_JOINT_NAMES),
        state_joints_spec="qpos(joints)",
        state_ee_spec="ee_pos(3) + ee_quat_wxyz(4)",
        state_cube_spec="not_stored_in_multicube_raw",
        pos_cube_red_spec="red_cube_pos(3) + red_cube_quat_wxyz(4)",
        pos_cube_green_spec="green_cube_pos(3) + green_cube_quat_wxyz(4)",
        pos_cube_blue_spec="blue_cube_pos(3) + blue_cube_quat_wxyz(4)",
        state_goal_spec="one_hot(red, green, blue) = 3",
        goal_pos_spec="bin_center_world_xyz(3)",
        state_gripper_spec="gripper_angle(1)",
        action_gripper_spec="gripper_ctrl(1)",
        control_hz=float(args.control_hz),
        cameras_display=list(CAMERA_NAMES),
    )

    # ── environment ───────────────────────────────────────────────────
    env = SO100MulticubeSimEnv(
        xml_path=xml_path,
        control_hz=args.control_hz,
        render_w=args.render_w,
        render_h=args.render_h,
        use_mocap=True,
        goal_cube="red",
        shuffle_cubes=not args.no_shuffle,
        seed=args.seed,
    )

    # ── collection loop ───────────────────────────────────────────────
    window_name = "Scripted Multicube Collection"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    successes = 0
    attempts = 0
    fail_counts: dict[str, int] = {
        "timeout": 0,
        "out_of_bounds": 0,
        "wrong_cube": 0,
        "user_abort": 0,
        "too_short": 0,
    }
    schedule_idx = 0

    print(f"\nScripted multicube collection")
    print(f"  target episodes : {args.num_episodes}")
    print(f"  max_steps       : {args.max_steps}")
    print(f"  post_success    : {args.post_success_steps}")
    print(f"  headless        : {args.headless}")
    print(f"  goal_cube       : {args.goal_cube}")
    print(f"  goal_sampling   : {args.goal_sampling}")
    print(f"  seed            : {args.seed}")
    print(f"  output          : {out_zarr}")
    print()

    try:
        while successes < args.num_episodes:
            if schedule_idx >= len(goal_schedule):
                extra = build_goal_schedule(
                    args.goal_cube, args.num_episodes, args.goal_sampling, rng
                )
                goal_schedule.extend(extra)
            goal_color = goal_schedule[schedule_idx]
            schedule_idx += 1
            attempts += 1

            steps_before = writer.num_steps_total

            ok, reason = run_scripted_episode(
                env,
                writer,
                goal_color,
                max_steps=args.max_steps,
                post_success_steps=args.post_success_steps,
                headless=args.headless,
                ep_idx=attempts,
                successes_so_far=successes,
                total_so_far=attempts - 1,
            )

            ep_len = writer.num_steps_total - steps_before

            if reason == "user_abort":
                fail_counts["user_abort"] += 1
                writer.discard_episode()
                print("Aborted by user.")
                break

            if ok and ep_len >= args.min_steps_per_episode:
                writer.end_episode()
                successes += 1
                if args.headless and successes % 10 == 0:
                    print(
                        f"  [{successes}/{args.num_episodes}] "
                        f"saved ({attempts} attempts, "
                        f"{attempts - successes} discarded)"
                    )
                elif not args.headless:
                    print(f"  ep {attempts}: SUCCESS ({goal_color}, {ep_len} steps)")
            else:
                writer.discard_episode()
                if ok and ep_len < args.min_steps_per_episode:
                    fail_counts["too_short"] += 1
                    tag = "too_short"
                else:
                    tag = reason or "unknown"
                    if tag in fail_counts:
                        fail_counts[tag] += 1

                if not args.headless:
                    print(f"  ep {attempts}: FAIL ({goal_color}, reason={tag}, {ep_len} steps)")

    finally:
        writer.flush()
        if not args.headless:
            cv2.destroyAllWindows()

    total_failures = sum(fail_counts.values())
    print(f"\n{'=' * 55}")
    print(f"  Collection complete")
    print(f"  Saved episodes  : {successes}")
    print(f"  Total attempts  : {attempts}")
    print(f"  Discarded       : {total_failures}")
    if total_failures > 0:
        for reason, count in fail_counts.items():
            if count > 0:
                print(f"    {reason:15s}: {count}")
    print(f"  Output          : {out_zarr}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
