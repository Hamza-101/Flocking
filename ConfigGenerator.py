import random
import json
import os
import numpy as np
import matplotlib.pyplot as plt


CartesianLimits={"X" : 5, "Y" : 5}
DataParams={"safe_distance": 1, "Neighborhood": 3 }
InitParams={"X_Max": CartesianLimits["X"], "X_Min": -CartesianLimits["X"],
             "Y_Max": CartesianLimits["Y"],"Y_Min": -CartesianLimits["Y"], 
             "NumAgents": 15}

# -(CartesianLimits["X"]
#For generating new filenames

TotalConfigs = 0

class Agent:
    def __init__(self):
        self.position = np.array([(random.uniform(round(InitParams["X_Min"], 1), round(InitParams["X_Max"], 1))), random.uniform(round(InitParams["Y_Min"], 1), round(InitParams["Y_Max"], 1))])                           
        
    #Get Connectedness
    def get_neighbors(agent, agents, radius):
        neighborhood=False
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
    global TotalConfigs  # Declare TotalConfigs as global

    folder_name = "CorrectConfigs"

    if(ConfigValidation==False):
        folder_name="IncorrectConfigs"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    while(os.path.exists(f"{folder_name}/Config_{TotalConfigs}.json")==True):
       TotalConfigs = TotalConfigs + 1

    filename = f"{folder_name}/Config_{TotalConfigs}.json"

    with open(filename, "w") as f:
        json.dump(agent_locations, f, cls=Encoder)  # Use the custom encoder
        f.write("\n")

    return folder_name

def property_check(agents):

    # Check safety and connectedness for each agent
    for agent in agents:
        # Safe distance check
        if (Agent.check_min_distance(agent, agents, DataParams["safe_distance"])==True):
                return False

        # Connectedness check
        if (Agent.get_neighbors(agent, agents, DataParams["Neighborhood"])==False):
                return False

    # Return True if all agents pass both safety and connectedness checks
    return True

def plot_agents(agents, folder_name):
    # Extract x and y coordinates of agents' positions
    x_coordinates = [agent.position[0] for agent in agents]
    y_coordinates = [agent.position[1] for agent in agents]

    # Set up the scatter plot
    plt.figure(figsize=(CartesianLimits["X"] + 5, CartesianLimits["Y"] + 5))
    plt.scatter(x_coordinates, y_coordinates, c='red', label='Agents', marker='o')
    
    # Set plot labels and title
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')

    PlotTitle = folder_name.replace('s','')
    plt.title(PlotTitle)

    # Set equal aspect ratio for axis scaling (1:1)
    plt.axis("equal")

    # Set plot limits based on CartesianLimits
    plt.xlim(InitParams["X_Min"], InitParams["X_Max"])
    plt.ylim(InitParams["Y_Min"], InitParams["Y_Max"])

    # Display the plot
    plt.legend()
    plt.grid(True)
    plt.show()


agents = [Agent() for _ in range(InitParams["NumAgents"])]

check_result = property_check(agents)
AgentsData = get_agent_locations(agents)
folder_name = save_config(check_result, AgentsData)

plot_agents(agents, folder_name)