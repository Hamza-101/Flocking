SimulationVariables = {
    "SimAgents": 20,
    "AgentData": [],
    "SafetyRadius": 2, 
    "NeighborhoodRadius": 10,
    "VelocityUpperLimit": 5.0,
    "VelocityInit": 1.0,
    "AccelerationUpperLimit": 1.0,
    "AccelerationInit": 0.0,
    "dt" : 0.1,
    "TimeSteps": 1000,
    "Runtime": 100,
    "X" : 10,
    "Y" : 10
}

ReynoldsVariables = {
    "w_alignment" : 0.5,
    "w_cohesion" : 1.5,
    "w_separation" : 3.0
}

Results = {
    "Directory": "Simulations",
    "Sim": "Simulation",
    "InitPositions": "Simulations\Config_0",
    "SimDetails": SimulationVariables
}
