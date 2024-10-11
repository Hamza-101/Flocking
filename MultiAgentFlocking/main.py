import gymnasium as gym
import json
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from Parameters import *
from gymnasium import spaces
from Agent import Agent
th.cuda.is_available = lambda: True

# Custom actor network per agent

class CustomActor(th.nn.Module):
    def __init__(self, observation_space, action_space):
        super(CustomActor, self).__init__()

        # Create 8 layers with 512 neurons each
        self.layers = th.nn.ModuleList()
        input_size = observation_space.shape[0]

        for _ in range(8):
            self.layers.append(th.nn.Linear(input_size, 512))
            input_size = 512  # Update input size for the next layer

        # Update action head based on action space type
        if isinstance(action_space, spaces.Box):  # Continuous action space
            self.action_head = th.nn.Linear(512, action_space.shape[0])
        elif isinstance(action_space, spaces.Discrete):  # Discrete action space
            self.action_head = th.nn.Linear(512, action_space.n)
        else:
            raise NotImplementedError("Action space type not supported")

    def forward(self, x):
        # Convert input to torch tensor if it's a numpy array
        if isinstance(x, np.ndarray):
            x = th.tensor(x, dtype=th.float32)
            
        for layer in self.layers:
            x = F.relu(layer(x))
        action_logits = self.action_head(x)
        return action_logits

class SharedCritic(th.nn.Module):
    def __init__(self, observation_space):
        super(SharedCritic, self).__init__()

        # Create 8 layers with 512 neurons each
        self.layers = th.nn.ModuleList()
        input_size = observation_space.shape[0]

        for _ in range(8):
            self.layers.append(th.nn.Linear(input_size, 512))
            input_size = 512  # Update input size for the next layer

        self.value_head = th.nn.Linear(512, 1)

    def forward(self, x):
        # Convert input to torch tensor if it's a numpy array
        if isinstance(x, np.ndarray):
            x = th.tensor(x, dtype=th.float32)

        for layer in self.layers:
            x = F.relu(layer(x))
        value = self.value_head(x)
        return value


# Custom Actor-Critic policy for PPO with different actors per agent and shared critic
class CustomMultiAgentPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, use_sde=False, **kwargs):
        super(CustomMultiAgentPolicy, self).__init__(observation_space, action_space, lr_schedule, use_sde=use_sde, **kwargs)
        
        # Create different actors for each agent (assuming max 3 agents here, adjust as necessary)
        self.actors = [CustomActor(observation_space, action_space) for _ in range(3)]

        # Shared critic
        self.critic = SharedCritic(observation_space)

    def forward(self, obs, agent_id):
        if agent_id < len(self.actors):
            action_logits = self.actors[agent_id](obs)
        else:
            raise ValueError("Agent ID out of range")
        return action_logits

    def critic_forward(self, obs):
        return self.critic(obs)
        
class ReplayBuffer:
    def __init__(self, buffer_size, num_agents, action_dim, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.state_dim = num_agents * 4  # 4-dimensional observation per agent (position and velocity)
        self.action_dim = action_dim

        # Preallocate memory for experience tuples
        self.states = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.next_states = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, 1), dtype=np.float32)

        self.position = 0
        self.size = 0  # Keeps track of the current number of stored transitions

    def add(self, state, action, reward, next_state, done):
        """
        Add experience tuple (state, action, reward, next_state, done) to the buffer.
        This overwrites the oldest data when the buffer is full.
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        # Update the position and size
        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        """
        Randomly sample a batch of experiences from the buffer.
        """
        indices = np.random.randint(0, self.size, size=self.batch_size)

        batch = dict(
            states=self.states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices],
            dones=self.dones[indices],
        )

        return batch

    def clear(self):
        """
        Reset the buffer.
        """
        self.position = 0
        self.size = 0
        self.states.fill(0)
        self.actions.fill(0)
        self.rewards.fill(0)
        self.next_states.fill(0)
        self.dones.fill(0)

    def is_ready(self):
        """
        Check if the buffer contains enough samples for training.
        """
        return self.size >= self.batch_size
    
    def save_buffer(self, replay_buffer, file_path):
        np.savez_compressed(file_path,
                            states=replay_buffer.states,
                            actions=replay_buffer.actions,
                            rewards=replay_buffer.rewards,
                            next_states=replay_buffer.next_states,
                            dones=replay_buffer.dones,
                            position=replay_buffer.position,
                            size=replay_buffer.size)
        print(f"Replay buffer saved to {file_path}")

    def load_buffer(self, file_path, replay_buffer):
        data = np.load(file_path)
        
        replay_buffer.states = data['states']
        replay_buffer.actions = data['actions']
        replay_buffer.rewards = data['rewards']
        replay_buffer.next_states = data['next_states']
        replay_buffer.dones = data['dones']
        replay_buffer.position = data['position']
        replay_buffer.size = data['size']
        
        print(f"Replay buffer loaded from {file_path}")

#Multiple random initialization

class Encoder(json.JSONEncoder):
    def default(self, obj):
        return json.JSONEncoder.default(self, obj)  

# 3 Agents
class FlockingEnv(gym.Env):
    def __init__(self):
        super(FlockingEnv, self).__init__()
        self.episode=0
        self.counter=3602
        self.CTDE=False
        self.current_timestep = 0
        self.reward_log = []
        self.np_random, _ = gym.utils.seeding.np_random(None)

        self.agents = [Agent(position) for position in self.read_agent_locations()]

        # Use settings file in actions and observations
        min_action = np.array([-5, -5] * len(self.agents), dtype=np.float32)
        max_action = np.array([5, 5] * len(self.agents), dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

        #Check this
        min_obs = np.array([-np.inf, -np.inf, -2.5, -2.5] * len(self.agents), dtype=np.float32)
        max_obs = np.array([np.inf, np.inf, 2.5, 2.5] * len(self.agents), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def step(self, actions):

        training_rewards = {}
        
        #REM
        noisy_actions = actions + np.random.normal(loc=0, scale=0.5, size=actions.shape)
        actions = np.clip(noisy_actions, self.action_space.low, self.action_space.high)

        self.current_timestep += 1
        reward=0
        done=False
        info={}
        
        #Noisy Actions
        observations = self.simulate_agents(actions)
        reward, out_of_flock = self.calculate_reward()

        if (self.CTDE==False):
            for agent in self.agents:
                if((self.check_collision(agent)) or (out_of_flock==True)):
                    done=True
                    env.reset()

        #Check position
        with open("training_rewards.json", "w") as f:
            json.dump(training_rewards, f)

        self.current_timestep = self.current_timestep + 1

        return observations, reward, done, info

    def reset(self, seed=None):
        # If a seed is provided, set it here
        if seed is not None:
            self.seed(seed)

        self.agents = [Agent(position) for position in self.read_agent_locations()]

        for agent in self.agents:
            agent.acceleration = np.zeros(2)
            agent.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"], size=2), 2)

        observation = self.get_observation().flatten()

        ################################
        self.current_timestep = 0  # Reset time step count
        ################################
        
        return observation

    def close(self):
        print("Environment is closed. Cleanup complete.")
        
    def simulate_agents(self, actions):

        observations = []  # Initialize an empty 1D array

        actions_reshaped = actions.reshape(((SimulationVariables["SimAgents"]), 2))

        for i, agent in enumerate(self.agents):
            position, velocity = agent.update(actions_reshaped[i])
            observation_pair = np.concatenate([position, velocity])
            observations = np.concatenate([observations, observation_pair])  # Concatenate each pair directly

        return observations
    
    def check_collision(self, agent):

        for other in self.agents:
            if agent != other:
                distance = np.linalg.norm(agent.position - other.position)
                if distance < SimulationVariables["SafetyRadius"]:
                    return True
                
        return False

    def get_observation(self):
        observations = np.zeros((len(self.agents), 4), dtype=np.float32)
        
        for i, agent in enumerate(self.agents):
            observations[i] = [
                agent.position[0],
                agent.position[1],
                agent.velocity[0],
                agent.velocity[1]
            ]

        # Reshape the observation into 1D                    
        return observations
   
    def get_closest_neighbors(self, agent):

        neighbor_positions=[]
        neighbor_velocities=[]

        for _, other in enumerate(self.agents):
            if agent != other:
                distance = np.linalg.norm(other.position - agent.position)

                if(self.CTDE == True):

                    ################################################################
                    # if distance < SimulationVariables["NeighborhoodRadius"]:
                    #    neighbor_positions.append(other.position)
                    #    neighbor_velocities.append(other.velocity)
                    ################################################################
                    neighbor_positions.append(other.position)
                    neighbor_velocities.append(other.velocity)

                else:  
                    neighbor_positions.append(other.position)
                    neighbor_velocities.append(other.velocity)

        return neighbor_positions, neighbor_velocities         
   
    def calculate_reward(self):
        reward=0
        Collisions={}

        neighbor_positions = [[] for _ in range(len(self.agents))] 
        neighbor_velocities = [[] for _ in range(len(self.agents))]
        out_of_flock=False

        for idx, _ in enumerate(self.agents):
            Collisions[idx] = []

        for _, agent in enumerate(self.agents): 
            neighbor_positions, neighbor_velocities=self.get_closest_neighbors(agent)
            val, out_of_flock=self.reward(agent, neighbor_velocities, neighbor_positions)
            reward+=val

        return reward, out_of_flock

    def reward(self, agent, neighbor_velocities, neighbor_positions):
        CohesionReward = 0
        AlignmentReward = 0
        total_reward = 0
        outofflock = False
        midpoint = (SimulationVariables["SafetyRadius"] + SimulationVariables["NeighborhoodRadius"]) / 2

        if len(neighbor_positions) > 0:
            for neighbor_position in neighbor_positions:
                distance = np.linalg.norm(agent.position - neighbor_position)

                if distance <= SimulationVariables["SafetyRadius"]:
                    CohesionReward += 0
                    
                elif SimulationVariables["SafetyRadius"] < distance <= midpoint:
                    CohesionReward += (10 / (midpoint - SimulationVariables["SafetyRadius"])) * (distance - SimulationVariables["SafetyRadius"])

                elif midpoint < distance <= SimulationVariables["NeighborhoodRadius"]:
                    CohesionReward += 10 - (10 / (SimulationVariables["NeighborhoodRadius"] - midpoint)) * (distance - midpoint)
      
                average_velocity = np.mean(neighbor_velocities, axis = 0)
                dot_product = np.dot(average_velocity, agent.velocity)
                norm_product = np.linalg.norm(average_velocity) * np.linalg.norm(agent.velocity)

                if norm_product == 0:
                    cos_angle = 1.0
                else:
                    cos_angle = dot_product / norm_product

                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                orientation_diff = np.arccos(cos_angle)


                alignment = (orientation_diff / np.pi)
                AlignmentReward = -20 * alignment + 10  

        else:
            CohesionReward -= 10
            outofflock = True

        total_reward = CohesionReward + AlignmentReward

        return total_reward, outofflock

    def read_agent_locations(self):

        File = rf"{Results['InitPositions']}" + str(self.counter) + "\config.json"
        with open(File, "r") as f:
            data = json.load(f)

        return data

    def seed(self, seed=SimulationVariables["Seed"]):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]

class BaselineController:
    def __init__(self):
        # No new attributes are initialized
        pass

    def flock(self, agent, observations):
        # Extract positions and velocities from observations
        positions = observations['positions']
        velocities = observations['velocities']
        neighbor_indices = env.get_closest_neighbors(agent)

        # Compute forces based on neighbors
        alignment = self.align(neighbor_indices, velocities)
        cohesion = self.cohere(neighbor_indices, positions)
        separation = self.separate(neighbor_indices, positions)

        total_force = (
            ((SimulationVariables["w_alignment"]) * alignment) +
            ((SimulationVariables["w_cohesion"]) * cohesion) +
            ((SimulationVariables["w_separation"]) * separation)
        )

        # Assuming max_acceleration is defined in SimulationVariables
        max_acceleration = SimulationVariables.get("max_acceleration", 1.0)  # Default if not set
        # Update acceleration with the computed forces
        acceleration = np.clip(total_force, -max_acceleration, max_acceleration)

        # Return the acceleration as the agent's decision value
        return acceleration

    def align(self, neighbor_indices, velocities):
        if len(neighbor_indices) > 0:
            neighbor_velocities = velocities[neighbor_indices]
            average_velocity = np.mean(neighbor_velocities, axis=0)
            # Assume a default self.velocity, as it's not initialized in __init__
            self.velocity = np.zeros(2)  # Placeholder, needs actual implementation
            desired_velocity = average_velocity - self.velocity
            return desired_velocity
        else:
            return np.zeros(2)

    def cohere(self, neighbor_indices, positions):
        if len(neighbor_indices) > 0:
            neighbor_positions = positions[neighbor_indices]
            # Assume a default self.position, as it's not initialized in __init__
            self.position = np.zeros(2)  # Placeholder, needs actual implementation
            center_of_mass = np.mean(neighbor_positions, axis=0)
            desired_direction = center_of_mass - self.position
            return desired_direction
        else:
            return np.zeros(2)

    def separate(self, neighbor_indices, positions):
        if len(neighbor_indices) > 0:
            neighbor_positions = positions[neighbor_indices]
            separation_force = np.zeros(2)

            # Assume a default self.position, as it's not initialized in __init__
            self.position = np.zeros(2)  # Placeholder, needs actual implementation

            for neighbor_position in neighbor_positions:
                relative_position = self.position - neighbor_position
                distance = np.linalg.norm(relative_position)

                if distance > 0:
                    separation_force += (relative_position / (distance * distance))

            return separation_force
        else:
            return np.zeros(2)























