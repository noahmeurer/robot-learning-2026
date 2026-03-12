# Exercise 2: Theoretical Questions

1. If you increase $K_P$ too much, the robot will overshoot the target and the tracking becomes very unstable.
2. The damping gain $K_D$ helps to mitigates that overshooting by penalizing large velocities as the error shrinks towards zero, thus applying a braking force.
3. Non-zero $K_I$ is needed when you have constant external forces such as gravity acting on the robot, otherwise you get steady-state error and the robot won't converge fully to the target.