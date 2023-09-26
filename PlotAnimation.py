from Params import *
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy  as np

data_file = "all_agents_data.json"

fig, ax = plt.subplots()

def read_data(data_file):
    with open(data_file, "r") as f:
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

            # Add lines showing velocity and acceleration
            velocity = data_point.get("Velocity", [0, 0])
            # acceleration = data_point.get("Acceleration", [0, 0])

            # Calculate velocity magnitude
            velocity_magnitude = np.linalg.norm(velocity)

            # Check if velocity_magnitude is not zero
            if velocity_magnitude > 0:
                # Normalize velocity vector
                normalized_velocity = velocity / velocity_magnitude

                # Draw velocity arrow (green)
                plt.arrow(x_pos, y_pos, normalized_velocity[0], normalized_velocity[1], head_width=0.2, head_length=0.2, fc='g', ec='g')

            # # Draw acceleration line (blue)
            # plt.arrow(x_pos, y_pos, acceleration[0], acceleration[1], head_width=0.2, head_length=0.3, fc='b', ec='b')

    plt.grid(True)


agent_data, max_num_points = read_data(data_file)
anim = animation.FuncAnimation(fig, function, fargs=(agent_data,), frames=max_num_points, interval=100, blit=False)
anim.save(animation_filename, writer="ffmpeg")
plt.show()