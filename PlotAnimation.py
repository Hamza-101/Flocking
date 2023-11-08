import json
from tqdm import tqdm
from Params import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

fig, ax = plt.subplots()

def read_data():
    with open(rf"{Results['Positions']}", "r") as f:
        all_data = json.load(f)
    agent_data = {}

    for agent_id, agent_points in all_data.items():
        agent_data[int(agent_id)] = agent_points

    max_num_points = max(len(agent_points) for agent_points in agent_data.values())

    return agent_data, max_num_points

def function(i, agent_data):
    plt.clf()
    plt.cla()

    plt.axis('equal')  

    for agent_id, data_points in agent_data.items():
        if i < len(data_points):
            data_point = data_points[i]
            x_pos = data_point[0]  
            y_pos = data_point[1]  

            plt.scatter(x_pos, y_pos, c='r', marker='o', s=20)

            circle = Circle((x_pos, y_pos), radius=5, edgecolor='black', facecolor='none', linewidth=0.1)
            plt.gca().add_patch(circle)

    plt.grid(True)

agent_data, max_num_points = read_data()
num_timesteps = 1000  


for i in tqdm(range(num_timesteps)):
    function(i, agent_data)

anim = animation.FuncAnimation(fig, function, fargs=(agent_data,), frames=num_timesteps, interval=100, blit=False)
anim.save(rf"{Results['Sim']}.mp4", writer="ffmpeg")
