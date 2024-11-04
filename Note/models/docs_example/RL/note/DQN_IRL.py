import tensorflow as tf
from Note import nn
import gym

class RewardNet(nn.Model):
    """Reward Network to approximate reward function in IRL."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.dense1 = nn.dense(64, state_dim + action_dim, activation='relu')
        self.dense2 = nn.dense(32, 64, activation='relu')
        self.output_layer = nn.dense(1, 32, activation='sigmoid')

    def __call__(self, state_action):
        x = self.dense1(state_action)
        x = self.dense2(x)
        reward = self.output_layer(x)
        return reward

class Qnet(nn.Model):
    """Q-network for estimating Q-values in RL."""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.dense1 = nn.dense(hidden_dim, state_dim, activation='relu')
        self.dense2 = nn.dense(action_dim, hidden_dim)
    
    def __call__(self, x):
        x = self.dense2(self.dense1(x))
        return x

class Env:
    """Custom environment with access to expert trajectories."""
    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.state = None

    def reset(self, seed=None):
        self.state = self.env.reset(seed=seed)
        self.expert_state = self.state  # Load expert's initial state here
        return self.state

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        self.expert_state = next_state
        self.state = next_state
        return [self.expert_state, self.state], reward, done, None

class DQN(nn.RL):
    """DQN agent that uses IRL reward estimation."""
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.q_net = Qnet(state_dim, hidden_dim, action_dim)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim)
        self.reward_net = RewardNet(state_dim, action_dim)
        self.param = [self.q_net.param,self.reward_net.param]
        self.env = Env()
    
    def action(self, s):
        """Return both expert action and agent's action."""
        expert_action = 0  # Placeholder, replace with actual expert action logic
        agent_action = self.q_net(s)
        return [expert_action, agent_action]
    
    def compute_reward(self, state, action):
        """Compute reward using the RewardNet."""
        state_action = tf.concat([state, action], axis=1)
        reward = self.reward_net(state_action)
        return tf.squeeze(reward, axis=-1)
    
    def __call__(self, s, a, next_s, d):
        """Compute loss using Q-values and IRL-generated rewards."""
        # Expand action dimensions to match Q-net input shape
        a = tf.expand_dims(a[:,1], axis=1)
        
        # Get Q-value for current state-action pair
        q_value = tf.gather(self.q_net(s[:,1]), a[:,1], axis=1, batch_dims=1)
        next_q_value = tf.reduce_max(self.target_q_net(next_s[:,1]), axis=1)
        
        # Compute rewards for both expert and agent trajectories
        expert_reward = self.compute_reward(s[:,0], a[:,0])
        agent_reward = self.compute_reward(s[:,1], a[:,1])
        
        # Compute TD target
        target = tf.cast(agent_reward, 'float32') + 0.98 * next_q_value * (1 - tf.cast(d, 'float32'))
        TD = q_value - target
        
        # Define IRL loss as the difference between expert and agent rewards
        irl_loss = tf.maximum(0.0, 1+agent_reward - expert_reward)
        
        # Combine TD loss and IRL loss
        q_loss = tf.reduce_mean(TD ** 2)
        
        return [q_loss, irl_loss]
    
    def update_param(self):
        """Update target Q-net parameters."""
        nn.assign_param(self.target_q_net.param, self.param)
