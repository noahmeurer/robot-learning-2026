# Homework 2 Deliverables Checklist

## Global Submission Requirements

- [x] Solved individually (no shared code).
- [x] All required files are ready for Gradescope upload.
- [x] Videos are in `.mp4` format.
- [x] Total required video content across Exercises 1-3 is under the global cap:
  - [x] `< 5:30` total (or `< 4:30` if not doing bonus).
- [x] If total upload size exceeds `100 MB`, split into multiple smaller video files.

## Exercise 1: IK and Lemniscate Tracking

### Code
- [x] Completed all TODOs in `exercises/ex1.py`:
  - [x] `get_lemniscate_keypoint(t, a)`
  - [x] `build_keypoints(count, width, x_offset, z_offset)`
  - [x] IK TODOs in `ik_track`

### Video (Required)
- [x] `.mp4` video showing robot tracking generated keypoints.
- [x] Includes answers to all Exercise 1 theoretical questions.
- [x] Video duration `< 2:00` including motion + theory answers.

### Theory
- [x] 1-sentence answer: effect of increasing lemniscate width `a`.
- [x] 1-sentence answer: effect of changing IK `dt`.
- [x] 1-sentence answer: numerical IK vs analytical IK (pros/cons).
- [x] 1-sentence answer: limits of this IK solver vs SOTA solvers.

## Exercise 2: Quintic Waypoints and PID

### Code
- [x] Completed all TODOs in `exercises/ex2.py`:
  - [x] `generate_quintic_spline_waypoints(start, end, num_points)`
  - [x] PID terms in `pid_control(...)`

### Video (Required)
- [x] `.mp4` video showing robot moving between waypoints.
- [x] Includes answers to all Exercise 2 theoretical questions.
- [x] Video duration `< 2:00` including motion + theory answers.

### Theory
- [x] 1-sentence answer: what happens as `K_P` increases.
- [x] 1-sentence answer: how `K_D` mitigates that effect.
- [x] 1-sentence answer: when non-zero `K_I` is needed.

## Exercise 3: RL Policy for Random Targets

### Code
- [x] Completed all TODOs in `exercises/ex3.py`:
  - [x] reset robot
  - [x] sample/reset target position
  - [x] process action to joint targets
  - [x] compute reward
  - [x] build observation vector

### Video (Required)
- [x] `.mp4` video showing random-target tracking.
- [x] Includes terminal printouts of final `ee_tracking_error` and average over 10 episodes.
- [x] Required part duration `< 0:30`.
- [x] Average final EE tracking error `< 0.05` (target for full score).

### Bonus (Optional)
- [ ] If doing bonus: include theory answer + modified policy/environment performance.
- [ ] Bonus video segment `< 1:00`.
- [ ] Bonus code changes are included in submitted code.

## Final Upload Checklist

- [x] Code files included: `exercises/ex1.py`, `exercises/ex2.py`, `exercises/ex3.py`.
- [x] Video(s) include all required content across exercises.
- [x] Combined video runtime satisfies global limit (`< 5:30`, or `< 4:30` without bonus).
- [x] If needed, videos split only for file-size reasons (`> 100 MB`).
