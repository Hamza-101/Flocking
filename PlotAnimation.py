import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

parameters = {"folder_name": "animations", 
              "filename": "all_agents_data.json"}   



animation_filename = "scatter_animation.mp4"

fig, ax = plt.subplots()

def read_data(data_file):

    with open(data_file, "r") as f:
        all_data = json.load(f)

    # Create an empty DataFrame to store agent data
    data = pd.DataFrame(columns=["AgentID","timestep", "X", "Y"])

    agentlist=[]

    for agent_id, agent_data in all_data.items():
        agentlist.append(int(agent_id))
        step = 0
        for data_point in agent_data:
            x_pos = data_point["x"]
            y_pos = data_point["y"]
            data = data._append({"AgentID": int(agent_id), "timestep": int(step), "X": x_pos, "Y": y_pos}, ignore_index=True)
            step = step + 1       
    

    return step, agentlist, data

def function(i, agentlist, agents_coordinates):
    plt.clf()
    plt.cla()
    
    plt.axis('equal')  # Add this line to set equal aspect ratio

    for agent in agentlist:
        agent_data_at_timestep = agents_coordinates[(agents_coordinates["AgentID"] == agent) & (agents_coordinates["timestep"] == i)]
        plt.scatter(agent_data_at_timestep["X"], agent_data_at_timestep["Y"], c='r', marker='o', s=20)

        # Plot the data for the selected agent at timestep i
        if not agent_data_at_timestep.empty:
            agent_pos_x = agent_data_at_timestep["X"].values[0]
            agent_pos_y = agent_data_at_timestep["Y"].values[0]

            # Draw a circle around the agent's position
            circle = Circle((agent_pos_x, agent_pos_y), radius=5, edgecolor='black', facecolor='none', linewidth=0.1)  # Adjust linewidth here
            plt.gca().add_patch(circle)

            # # Draw the line emanating from the agent
            # line_length = 7  # Adjust the line length as needed
            # plt.arrow(agent_pos_x, agent_pos_y, line_length, 0, head_width=1, head_length=10, fc='red', ec='black', linewidth=1)

    # Calculate the maximum and minimum x and y positions of all agents at this timestep
    max_x = agents_coordinates[agents_coordinates["timestep"] == i]["X"].max()
    min_x = agents_coordinates[agents_coordinates["timestep"] == i]["X"].min()
    max_y = agents_coordinates[agents_coordinates["timestep"] == i]["Y"].max()
    min_y = agents_coordinates[agents_coordinates["timestep"] == i]["Y"].min()

    # Set the axis limits based on the maximum and minimum x and y positions
    plt.xlim(min_x - 25, max_x + 25)  # Adding some buffer space of 25 units
    plt.ylim(min_y - 25, max_y + 25)

    plt.title(f"Flocking")
    plt.grid(True)


step, agent_list, agents_coordinates = read_data(parameters["filename"])

anim = animation.FuncAnimation(fig, function, fargs=( agent_list, agents_coordinates,), frames=step, interval=167, blit=False)

# anim.save(animation_filename, writer="pillow", fps=30)
anim.save(animation_filename, writer="ffmpeg", fps=10)

print("file_saved")
plt.show()

