import random
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from Params import *

TotalConfigs = 0

class Agent:
    def __init__(self):
        self.position = np.array([round(random.uniform(InitParams["X_Min"], InitParams["X_Max"]), 1), round(random.uniform(InitParams["Y_Min"], InitParams["Y_Max"]), 1)])
        
    #Get Connectedness
    def get_neighbors(agent, agents, radius):
        neighbors = []
        
        for other_agent in agents:
            if agent != other_agent:
                distance = np.linalg.norm(agent.position - other_agent.position)
                if distance < radius:
                    neighbors.append(other_agent)
                    return True # Stop searching for more neighbors once at least one is found within the radius

        return False

    #Safety
    def check_min_distance(agent, agents, threshold):
        for other_agent in agents:
            if agent != other_agent:
                distance = np.linalg.norm(agent.position - other_agent.position)
                if distance < threshold:
                    return False

        return True

class Encoder(json.JSONEncoder):
    def default(self, obj):
        return json.JSONEncoder.default(self, obj)

def get_agent_locations(agents):
    agent_locations = [agent.position.tolist() for agent in agents]
    return agent_locations

def save_config(ConfigValidation, agent_locations):
    global TotalConfigs  

    folder_name = "CorrectConfigs"

    if ConfigValidation == False:
        folder_name = "IncorrectConfigs"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    while(os.path.exists(f"{folder_name}/Config_{TotalConfigs}.json") == True):
        TotalConfigs = TotalConfigs + 1

    filename = f"{folder_name}/Config_{TotalConfigs}.json"

    with open(filename, "w") as f:
        json.dump(agent_locations, f, cls=Encoder)  # Serialize agent_locations and write it to the file
        f.write("\n")

    return folder_name

    # config_info = {
    #     "CartesianLimits": CartesianLimits,
    #     "DataParams": DataParams,
    #     "InitParams": InitParams,
    #     "AgentLocations": agent_locations
    # }

# config_info,




def property_check(agents):
    # Create a dictionary to store neighbors for each agent
    agent_neighbors = {agent: [] for agent in agents}

    # Check safety and connectedness for each agent
    for agent in agents:
        # Safe distance check
        if not Agent.check_min_distance(agent, agents, DataParams["safe_distance"]):
            return False

        # Check connectedness within the neighborhood distance
        for other_agent in agents:
            if agent != other_agent:
                distance = np.linalg.norm(agent.position - other_agent.position)
                if distance < DataParams["Neighborhood"]:
                    agent_neighbors[agent].append(other_agent)

    # Check if there are no neighbors within the neighborhood distance for any agent
    if any(not neighbors for neighbors in agent_neighbors.values()):
        return False

    # Return True if all agents pass both safety and connectedness checks
    return True

def plot_agents(agents, folder_name):

    x_coordinates = [agent.position[0] for agent in agents]
    y_coordinates = [agent.position[1] for agent in agents]

    plt.figure(figsize=(CartesianLimits["X"] + 5, CartesianLimits["Y"] + 5))
    plt.scatter(x_coordinates, y_coordinates, c='red', label='Agents', marker='o')

    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')

    PlotTitle = folder_name.replace('s','')
    plt.title(PlotTitle)

    # Set equal aspect ratio for axis scaling (1:1)
    plt.axis("equal")
    
    plt.xlim(InitParams["X_Min"], InitParams["X_Max"])
    plt.ylim(InitParams["Y_Min"], InitParams["Y_Max"])

    plt.legend()
    plt.grid(True)
    plt.show()

i = 0
j = 0
while(i<=10000):
    agents = [Agent() for _ in range(InitParams["NumAgents"])]

    check_result = property_check(agents)
    AgentsData = get_agent_locations(agents)
    folder_name = save_config(check_result, AgentsData)
    if(check_result==True):
        print(check_result)
        j=j+1
    i = i + 1
print(j)    
    