1. Why is experience replay important in DQN?

- It breaks temporal correlations between consecutive transitions by instead sampling sampling randomly from a buffer of past experiences, which prevents the NN overfitting to recent events.
- Additionally, it improves data efficiencly by allows the network to learn multiple times from particularly informative experiences.


2. What is the role of the target network in DQN? How does it improve stability?

- Addresses the problem of chasing a moving target during gradient updates, helping to improve convergence. Without it, the target and the prediction would rely on the same network weights which destabilizes training. So instead, DQN uses a frozen target network that is only updated periodically.

3. What is Double DQN, and how does it reduce overestimation bias compared to standard DQN?

In DQN the parameterized predictions of the deep Q-net are inherently noisy. Because we use a max operator, the noise in predictions can act as a false signal, causing the agent to learn to favor actions with only artificially inflated Q-values - something refered to as overestimation bias. DDQN solves this by decoupling action selection from prediction - using the active network to select the best action, but using the frozen target network to evaluate that action's actual value, thereby reducing overestimation bias.