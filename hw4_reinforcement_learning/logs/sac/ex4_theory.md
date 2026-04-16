1. SAC adds an entropy bonus to the reward. What are the benefits of this?

The entropy bonus is added to the optimization objective to encourage a stochastic policy or, said differently, discourage the collapse of the policy to a more deterministic (or peaked) distribution. This can help ensure that the agent sufficiently explores it's environment to discover better actions and avoid getting stuck at poor local optima. Another benefit is improved training stability because the policy stays broader.

2. SAC squashes actions through tanh. Why does this require a log-probability correction?

Because tanh is a non-linear transformation, it doesn't preserve the probability density of the final action vs the raw action. To account for this, we have to apply the standard change-of-variables correction involving the Jacobian of the tanh function. This correction ensures that SAC makes un-biased estimates of how likely an action is after squashing, which is crucial to downstream loss computations.

3. The temperature (\alpha) is tuned automatically. What happens when the policy's entropy is above vs. below the target?

If the policy's entropy is above the target, then alpha is decreased. That weakens the entropy term, so the policy is pushed to become less random and focus more on reward maximization. If the policy entropy is below the target, then alpha is increased. That strengthens the entropy term, encouraging a more stochastic policy and more exploration.


4. How does SAC compare with PPO in terms of update-to-data (UTD) ratio? (UTD = gradient update steps / environment steps)
SAC UTD: = 200 / 500 = 0.4
PPO UTD: = 20 / 2048 = 0.00977 ≈ 0.01
Ratio = 0.4 / 0.00977 ≈ 41

This implies that SAC performs about 41x more gradient updates per environment step than PPO. This highlights the difference in the off-policy of SAC, which can re-use old-data to maximize sample efficiency, whereas PPO can only perform limited updates before the data it collected becomes stale and needs to be discarded.

5. Briefly discuss about the advantages and disadvantages of on-policy vs. off-policy algorithms.

Off-policy algorithms are more sample efficient because they can learn from diverse past experiences, even those collected by another policy (e.g., random exploration or expert demonstrations in robotics). The disadvantage is that off-policy algorithms may be more unstable during training and typically require more hyper-parameter tuning versus on-policy methods, which directly optimize the policy generating the data, but are less sample efficient as old data must be discarded.
