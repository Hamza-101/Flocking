import pygame
import random
import time
import json
import os
import numpy as np


# Initialize Pygame
pygame.init()

#PyGame Initialization
width, height = 100, 100
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
font = pygame.font.SysFont(None, 30)

agent_data=[]
num_agents = 15
SETTINGS={"NumAgents": 15, "AgentData": []}
DATAFILES={}



data_file = "all_agents_data.json"
file_path = "CorrectConfigs\Config_3.json"

if os.path.exists(data_file):
        os.remove(data_file)

# Define agent class
class Agent:

    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)], dtype=float)
        self.acceleration = np.array([0, 0], dtype=float)
        self.max_acceleration = 5
        self.max_velocity = 2.5     

    def update(self, agents, dt):
        # Convert position and velocity to NumPy arrays
        positions = np.array([agent.position for agent in agents])
        velocities = np.array([agent.velocity for agent in agents])

        #add neighbors number
        neighbor_indices = self.get_closest_neighbors(positions, 15)  # Increase the number of neighbors to 12 or higher
        self.flock(neighbor_indices, positions, velocities)

        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

    def flock(self, neighbor_indices, positions, velocities):

        w_alignment = 0.5  # Increase alignment weight
        w_cohesion = 1.5   # Decrease cohesion weight
        w_separation = 3
        
        alignment = self.align(neighbor_indices, velocities)
        cohesion = self.cohere(neighbor_indices, positions)
        separation = self.separate(neighbor_indices, positions)

        self.acceleration = (w_alignment * alignment) + (w_cohesion * cohesion) + (w_separation * separation)
        self.acceleration = np.clip(self.acceleration, 0, self.max_acceleration)

    def align(self, neighbor_indices, velocities):
        if len(neighbor_indices) > 0:
            neighbor_velocities = velocities[neighbor_indices]
            average_velocity = np.mean(neighbor_velocities, axis=0)
            alignment = average_velocity - self.velocity
            return alignment
        else:
            return np.zeros(2)

    def cohere(self, neighbor_indices, positions):
        if len(neighbor_indices) > 0:
            neighbor_positions = positions[neighbor_indices]
            center_of_mass = np.mean(neighbor_positions, axis=0)
            cohesion = center_of_mass - self.position
            return cohesion
        else:
            return np.zeros(2)

    def separate(self, neighbor_indices, positions):
        separation_radius = 20
        if len(neighbor_indices) > 0:
            neighbor_positions = positions[neighbor_indices]
            distances = np.linalg.norm(neighbor_positions - self.position, axis=1)
            valid_neighbors = neighbor_positions[distances > 0]
            num_valid_neighbors = len(valid_neighbors)
            if num_valid_neighbors > 0:
                separation_vector = np.sum(valid_neighbors - self.position, axis=0) / num_valid_neighbors
                separation_vector /= np.linalg.norm(separation_vector)
                return separation_vector
        return np.zeros(2)

    # def update(self, agents, dt):
    #     positions = np.array([agent.position for agent in agents])
    #     velocities = np.array([agent.velocity for agent in agents])

    #     neighbor_indices = self.get_closest_neighbors(agents, 5.0, max_neighbors = 5)  
    #     self.flock(neighbor_indices, positions, velocities)
    #     self.velocity += self.acceleration * dt
    #     self.position += self.velocity * dt

    # def flock(self, neighbor_indices, positions, velocities):

    #     w_alignment = 1
    #     w_cohesion = 3  
    #     w_separation =0
        
    #     alignment = self.align(neighbor_indices, velocities)
    #     cohesion = self.cohere(neighbor_indices, positions)
    #     separation = self.separate(neighbor_indices, positions)

    #     self.acceleration = (w_alignment * alignment) + (w_cohesion * cohesion) + (w_separation * separation)
    #     self.acceleration = np.clip(self.acceleration, 0, self.max_acceleration)

    # def align(self, neighbor_indices, velocities):
    #     if len(neighbor_indices) > 0:
    #         neighbor_velocities = velocities[neighbor_indices] 
    #         average_velocity = np.mean(neighbor_velocities, axis=0)
    #         alignment = average_velocity - self.velocity
    #         return alignment
    #     else:
    #         return np.zeros(2)

    # def cohere(self, neighbor_indices, positions):
    #     if len(neighbor_indices) > 0:
    #         neighbor_positions = positions[neighbor_indices]
    #         center_of_mass = np.mean(neighbor_positions, axis=0)
    #         cohesion = center_of_mass - self.position
    #         return cohesion
    #     else:
    #         return np.zeros(2)

    # def separate(self, neighbor_indices, positions):
    #     if len(neighbor_indices) > 0:
    #         neighbor_positions = positions[neighbor_indices]
    #         distances = np.linalg.norm(neighbor_positions - self.position, axis = 1)
    #         valid_neighbors = neighbor_positions[distances > 0]
    #         num_valid_neighbors = len(valid_neighbors)
    #         if num_valid_neighbors > 0:
    #             separation_vector = np.sum(valid_neighbors - self.position, axis=0) / num_valid_neighbors
    #             separation_vector /= np.linalg.norm(separation_vector)
    #             # print(separation_vector)
    #             return separation_vector
        
    #     return np.zeros(2)


        
    def get_closest_neighbors(self, positions, num_neighbors):
        distances = np.linalg.norm(positions - self.position, axis=1)
        sorted_indices = np.argsort(distances)
        closest_neighbors_indices = sorted_indices[1:num_neighbors + 1]  # Exclude self from neighbors
        return closest_neighbors_indices
 
    def draw(self):
        pygame.draw.circle(screen, WHITE, (int(self.position[0]), int(self.position[1])), 3)

class Encoder(json.JSONEncoder):
    def default(self, obj):
        return json.JSONEncoder.default(self, obj)

def read_agent_locations():
    with open(file_path, "r") as f:
        data = json.load(f)
        print("File loaded")
    return data

# Function to render the agents in a window
def render_agents(agents, wi, he):
    timer_start = time.time()
    total_runtime = 100
    sample_rate = 10  # Number of sample frames per second
    sample_interval = 1.0 / sample_rate  # Timestep
    time_elapsed = 0.0  # Time elapsed since the last sample
    agent_data = {id(agent): [] for agent in agents}  # Dictionary to store data for each agent
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_agent_data(agent_data, data_file)
                running = False
        
        # Calculate time step (dt)
        dt = clock.tick(60) / 1000.0
        
        # Check if any agent hits the boundary
        for agent in agents:
            agent_id=id(agent)
            agent.update(agents, dt)
            
            agent_data_full = {
                "x": round(agent.position[0], 2),  # Save x-coordinate
                "y": round(agent.position[1], 2)   # Save y-coordinate
                }
            
            time_elapsed += dt
            if time_elapsed >= sample_interval:
                agent_data[agent_id].append(agent_data_full)  # Append agent's data to the list
                time_elapsed = 0.0

            if running:
                # Clear the screen
                screen.fill(BLACK)
                # Draw agents
                for agent in agents:
                    agent.draw()
                # Display Number of agents
                count_label = font.render("Number of agents: {}".format(len(agents)), True, WHITE)
                screen.blit(count_label, (10, 10))
            
                pygame.display.flip()

        if (time.time() - timer_start >= total_runtime):
            running = False

    elapsed_time = time.time() - timer_start
    
    print("Time elapsed (seconds):", round(elapsed_time, 1))

    save_agent_data(agent_data, data_file)
    # Quit Pygame

    pygame.quit()

def save_agent_data(agent_data, data_file):
    if not os.path.exists(data_file):
        with open(data_file, "w") as f:
            json.dump({}, f)

    with open(data_file, "r") as f:
        all_data = json.load(f)

    for agent_id, data_list in agent_data.items():
        if agent_id not in all_data:
            all_data[agent_id] = []
        all_data[agent_id] += data_list

    with open(data_file, "w") as f:
        json.dump(all_data, f, cls=Encoder)

agent_locations = read_agent_locations()

agents = [Agent(position) for position in agent_locations]

#Creates new file everytime
render_agents(agents, width, height)
