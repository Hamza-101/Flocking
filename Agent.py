import pygame
import time
import json
import os
import numpy as np

pygame.init()

width, height = 700, 700
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
font = pygame.font.SysFont(None, 30)

from Params import *

class Agent:

    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array([-(SimulationVariables["VelocityInit"]), SimulationVariables["VelocityInit"]], dtype=float)
        self.acceleration = np.array([-(SimulationVariables["AccelerationInit"]), SimulationVariables["AccelerationInit"]], dtype=float)
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"] 
        self.max_velocity = SimulationVariables["VelocityUpperLimit"]  

    def update(self, agents, dt):
        positions = np.array([agent.position for agent in agents])
        velocities = np.array([agent.velocity for agent in agents])

        neighbor_indices = self.get_closest_neighbors(agents, SimulationVariables["NeighborhoodRadius"])
        self.flock(neighbor_indices, positions, velocities)
        self.velocity += self.acceleration * dt

        vel = np.linalg.norm(self.velocity)
        if vel > self.max_velocity:
            self.velocity = (self.velocity / vel) * self.max_velocity

        self.position += self.velocity * dt

    def flock(self, neighbor_indices, positions, velocities):

        alignment = self.align(neighbor_indices, velocities)
        cohesion = self.cohere(neighbor_indices, positions)
        separation = self.separate(neighbor_indices, positions)

        total_force = (
            ((ReynoldsVariables["w_alignment"])  * alignment) +
            ((ReynoldsVariables["w_cohesion"])  * cohesion) +
            ((ReynoldsVariables["w_separation"])  * separation)
        )

        self.acceleration = np.clip(total_force, 0, self.max_acceleration)   

    def align(self, neighbor_indices, velocities):
        if len(neighbor_indices) > 0:
            neighbor_velocities = velocities[neighbor_indices]
            average_velocity = np.mean(neighbor_velocities, axis = 0)
            desired_velocity = average_velocity - self.velocity
            return desired_velocity
        else:
            return np.zeros(2)

    def cohere(self, neighbor_indices, positions):
        if len(neighbor_indices) > 0:
            neighbor_positions = positions[neighbor_indices]
            center_of_mass = np.mean(neighbor_positions, axis = 0)
            desired_direction =  center_of_mass - self.position 
            return desired_direction
        else:
            return np.zeros(2)

    def separate(self, neighbor_indices, positions):
        if len(neighbor_indices) > 0:
            neighbor_positions = positions[neighbor_indices]
            separation_force = np.zeros(2)

            for neighbor_position in neighbor_positions:
                relative_position = self.position - neighbor_position
                distance = np.linalg.norm(relative_position)

                if distance > 0:
                    separation_force += (relative_position / (distance * distance))

            return separation_force
        else:
            return np.zeros(2)

    def get_closest_neighbors(self, agents, max_distance):

        neighbor_indices = []
  
        for i, agent in enumerate(agents):
            if agent != self:
                distance = np.linalg.norm(agent.position - self.position)
                if distance < max_distance:
                    neighbor_indices.append(i)

        return neighbor_indices 
 
    def draw(self):
        pygame.draw.circle(screen, WHITE, (int(self.position[0]), int(self.position[1])), 3)

class Encoder(json.JSONEncoder):
    def default(self, obj):
        return json.JSONEncoder.default(self, obj)

def read_agent_locations():
    with open(rf"{Results['InitPositions']}\config.json", "r") as f:
        data = json.load(f)
        print("File loaded")
    return data

def render_agents(agents, wi, he):
    timer_start = time.time()
    time_elapsed = 0.0  
    agent_data = {id(agent) : [] for agent in agents}   
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save_agent_data(agent_data, rf"{Results['InitPositions'] }\run.json")
                running = False

        for agent in agents:
            agent_id = id(agent)
            agent.update(agents, (SimulationVariables["dt"]))
            
            agent_data_full = {
                "x" : round(agent.position[0], 2), 
                "y" : round(agent.position[1], 2)
            }
            
            time_elapsed += (SimulationVariables["dt"])

            # Getting Samples
            if time_elapsed >= (SimulationVariables["dt"]): 
                agent_data[agent_id].append(agent_data_full)  
                time_elapsed = 0.0

        if running:
            screen.fill(BLACK)
            for agent in agents:
                agent.draw()
            count_label = font.render("Number of agents: {}".format(str(SimulationVariables["SimAgents"])), True, WHITE)
            screen.blit(count_label, (10, 10))
            
            pygame.display.flip()

        if (time.time() - timer_start >= (SimulationVariables["Runtime"])):
            running = False

    print("Simulation Complete")
    save_agent_data(agent_data)
    pygame.quit()

def save_agent_data(agent_data):
    if not os.path.exists(rf"{Results['InitPositions']}\run.json"):
        with open(rf"{Results['InitPositions']}\run.json", "w") as f:
            json.dump({}, f)

    if not os.path.exists(rf"{Results['InitPositions']}\settings.json"):
        with open(rf"{Results['InitPositions']}\settings.json", "w") as f:
            json.dumps(Results["SimDetails"], f)

    with open(rf"{Results['InitPositions']}\run.json", "r") as f:
        all_data = json.load(f)

    for agent_id, data_list in agent_data.items():
        if agent_id not in all_data:
            all_data[agent_id] = []
        all_data[agent_id] += data_list

    with open(rf"{Results['InitPositions']}\run.json", "w") as f:
        json.dump(all_data, f, cls=Encoder)




agent_locations = read_agent_locations()
agents = [Agent(position) for position in agent_locations]
render_agents(agents, width, height)
