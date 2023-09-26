#ConfigGenerator.py

#PlotAnimation.py




CartesianLimits={"X" : 20, "Y" : 20}
DataParams={"safe_distance": 2, "Neighborhood": 10}
InitParams={"X_Max": 10, "X_Min": -20,
             "Y_Max": 10,"Y_Min": -20, 
             "NumAgents": 30}

# PositionParams={"X" : 10, "Y" : 10}
# DataParams={"safe_distance": 2, "neighborhood_distance": 5}
# InitParams={"x_Max": 10, "x_Min": -0,
#              "y_Max": 10,"y_Min": -0, 
#              "num_agents": 15}

# "AgentData": []
# DataFile = "all_agents_data_config_"
# # InitFile = "CorrectConfigs\Config_25755.json"
# parameters = {"AnimationFolder": "animations", 
#               "SaveData": "all_agents_data.json"} 

# AnimationFile = "scatter_animation.mp4"

ModelParams = {
    "VelocityInit" : 1.0,
    "AccelerationInit" : 0.0,
    "AccelerationUL" : 1.0,
    "VelocityUL" : 5.0,
    "dt" : 0.1
}

ReynoldsParams = {
    "w_alignment" : 0.5,
    "w_cohesion" : 1.5,
    "w_separation" : 3.0
}


SETTINGS={"NumAgents": 20, "AgentData": []}
DATAFILES={}

#132923

data_file = "all_agents_data.json"
file_path = "CorrectConfigs\Config_132923.json"

agent_data=[]
num_agents = 20






parameters = {"folder_name": "animations", 
              "filename": "all_agents_data.json"}   

animation_filename = "scatter_animation.mp4"