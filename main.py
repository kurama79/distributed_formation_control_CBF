'''
    Formation algorithms:
        - Observers comparation
        - CBF proofs
        - Local control for reference tracking
'''

import numpy as np

from config import *
import plot_

def print_object_info(obj):
    
    '''
        Function generated to print the attributes and values of any object generated to know their
        current states or parameters.
            obj -> object to known its information.
    '''
    
    attributes = vars(obj)
    for attribute, value in attributes.items():
        print(attribute, "=", value)

# ___________________________________________ Main fuction________________________________________________ 
if __name__ == '__main__':

    # General variables
    agentsNumber = 1 # Number of agents in the network
    obstaclesNumber = 5 # Number of the obstacles in the environment
    CBFmethod = 10 # Here que define which technic of CBF use
    
    obstacles = generate_obstacles(obstaclesNumber, True)
    # obstacles = None
    L = generate_graph(agentsNumber, False) # Definig the Laplacian matrix
    leader, agents = generate_agents(agentsNumber, L, obstacles=obstacles, random=False, lPosition=[14, 19], CBFmethod=CBFmethod)
     
    # Define if the agents will use a oberver 
    for agent in agents:
        agent.set_observer_1st()
        agent.config_data_distances(agents)
        # agent.set_observer_2nd()

    t = 0.0 # Time
    time_data = []
    dt = 0.01 # Sample step
    simTime = 90
    while t <= simTime:# and not agentsQ[0].break_:
        # Main loop
            
        # leader.leader_dynamic(t, dt)
        leader.leader_fixed(t, dt)
        for agent in agents:
        #     # agent.observation(dt)
            agent.detect_neighbors(agents)
            
        for agent in agents:
            # agent.formation_control(dt, BFC=True)
            agent.formation_control_HO(dt)
        
        leader.save()
        for agent in agents:
            agent.save(agents)
            
        time_data.append(t)
        t += dt
    
    plot_.path(leader, agents, obst=obstacles)
    # plot_.variables_vs_time(leader, agents, time_data, simTime)