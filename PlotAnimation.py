from Params import *
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy  as np


fig, ax = plt.subplots()

def read_data():
    with open(rf"{Results['InitPositions']}\run.json", "r") as f:
        all_data = json.load(f)

    # Create a dictionary to store agent data
    agent_data = {}

    for agent_id, agent_points in all_data.items():
        agent_data[int(agent_id)] = agent_points

    # Find the maximum number of data points for any agent
    max_num_points = max(len(agent_points) for agent_points in agent_data.values())

    return agent_data, max_num_points

def function(i, agent_data):
    plt.clf()
    plt.cla()

    plt.axis('equal')  # Add this line to set equal aspect ratio

    for agent_id, data_points in agent_data.items():
        if i < len(data_points):
            data_point = data_points[i]
            x_pos = data_point["x"]
            y_pos = data_point["y"]

            # Plot the agent's data point
            plt.scatter(x_pos, y_pos, c = 'r', marker = 'o', s = 20)

            # Draw a circle around the agent's position
            circle = Circle((x_pos, y_pos), radius=5, edgecolor='black', facecolor='none', linewidth=0.1)
            plt.gca().add_patch(circle)

    plt.grid(True)



PositionsFile = Results["InitPositions"] + "\run.json"
agent_data, max_num_points = read_data()
anim = animation.FuncAnimation(fig, function, fargs = (agent_data,), frames = max_num_points, interval = 100, blit=False)
anim.save(rf"{Results['InitPositions']}\\{Results['Sim']}.mp4", writer="ffmpeg")
plt.show()
