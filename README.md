# one_step_actor_critic_mountain_car
Simple implementation of the one step actor-critic method in mountain car environment using discrete actions.

```python
# for each episode
for t in range(MAX_T):
            s = get_state(env)
            policy_prob = policy(env,s,theta)
            a = np.random.choice(range(env.action_space.n), p = policy_prob)
            _, reward, done, _ = env.step(a)
            s_next = get_state(env)
            delta = reward + value_approx(s_next,w) - value_approx(s,w)
            w = w + ALPHA_W * delta * value_approx_grad(s,w)
            theta = theta + ALPHA_THETA * delta * policy_grad(env,a,s,theta)
```


The following plot shows the rewards obtained in a run of the algorithm.
![Reward for episode with state dimension 200](https://raw.githubusercontent.com/gabrielesartor/one_step_actor_critic_mountain_car/master/mountain_car_actor_critic_s200_a3.png)
