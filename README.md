
# Reinforcement is All you need !

<!-- ## 1. REINFORCE Algorithm 

Objective:
$$
J(\theta) = \mathbb{E}_{\tau \sim p_{\theta}(\tau)} \left[ \sum_{t=1}^{T} r(s_t, a_t) \right]
$$

Policy Gradient:
$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_{\theta}} \left[ \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau) \right]
$$
where $R(\tau)$ is the total reward of trajectory $\tau$.

REINFORCE uses Monte Carlo sampling to estimate this gradient and update $\theta$. -->