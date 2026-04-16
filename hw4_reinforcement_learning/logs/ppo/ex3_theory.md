1. Why does PPO clip the probability ratio instead of directly constraining the KL divergence like TRPO? What goes wrong if you remove clipping entirely?

- TRPO enforces a hard KL constraint which require complex-to-implement and slower second-order optimization algorithms to keep updates within the "trust region". PPO uses clipping in a surrogate objective that implicitely constrains updates to this same "trust region", but can be easily implemented with first-order methods like gradient descent. If we removed clipping, then the optimizer would push the importance sampling ratios to extremes values where the probability distribution over actions is no longer valid, causing training instability.

2. PPO throws away all collected data after each update. Why can't you simply reuse old rollouts for more gradient steps?

- PPO is fundementally an on-policy algorithm. Although it reuses the current batch of data for a few epochs, it has to discard the data once the policy drifts too far from the one that collected it. Reusing old rollouts would cause the importance sampling ratios to have huge variance, making the training signal too noisy for the agent to converge.

3. What does the GAE parameter (\lambda) control? What happens at the extremes (\lambda = 0) and (\lambda = 1)?

- This trades-off bias vs. variance in the the advantage function estimate. When lambda = 0, we rely on the critic's value function to estimate the advantage, which has low variance but incurs high bias. When lambda = 1, we rely on the raw, noisy environmental rewards in our estimate, which incurs zero bias but high variance.