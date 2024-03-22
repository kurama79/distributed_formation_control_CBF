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
# def main():

    # General variables
    agentsNumber = 5 # Number of agents in the network
    obstaclesNumber = 8 # Number of the obstacles in the environment
    CBFmethod = 4 # Here que define which technic of CBF use
    
    obstacles = generate_obstacles(obstaclesNumber, True)
    # obstacles = None
    L = generate_graph(agentsNumber, False) # Definig the Laplacian matrix
    leader, agents = generate_agents(agentsNumber, L, obstacles=obstacles, random=False, lPosition=[14, 19], CBFmethod=CBFmethod)
    leaderQ, agentsQ = generate_agents(agentsNumber, L, obstacles=obstacles, random=False, lPosition=[14, 19], CBFmethod=6)
    
    # Set the same positions to compare performance
    for i, agent in enumerate(agents):
        agentsQ[i].x_ = agent.x_
        agentsQ[i].y_ = agent.y_
     
    # Define if the agents will use a oberver 
    for agent in agents:
        agent.set_observer_1st()
        agent.config_data_distances(agents)
        # agent.set_observer_2nd()
        
    for agent in agentsQ:
        agent.set_observer_1st()
        agent.config_data_distances(agents)
        # agent.set_observer_2nd()

    t = 0.0 # Time
    time_data = []
    dt = 0.01 # Sample step
    simTime = 150
    while t <= simTime:
        # Main loop
            
        leader.leader_dynamic(t, dt)
        leaderQ.leader_dynamic(t, dt)
        # leader.leader_fixed(t, dt)
        for agent in agents:
        #     # agent.observation(dt)
            agent.detect_neighbors(agents)
            
        # Compare agents
        for agent in agentsQ:
            agent.detect_neighbors(agents)
            
        for agent in agents:
            agent.formation_control(dt, BFC=True)
            # agent.formation_control_HO(dt)
            
        for agent in agentsQ:
            # agent.formation_control(dt, BFC=True)
            agent.formation_control_HO(dt)
        
        leader.save()
        leaderQ.save()
        for agent in agents:
            agent.save(agents)
            
        for agent in agentsQ:
            agent.save(agents)
        time_data.append(t)
        t += dt
        
    # # Show times
    # print('Generation time controlers, CBF-QP:')
    # for agent in agents:
    #     timeAverage = sum(agent.control_times_)/len(agent.control_times_)
    #     print('Agent ', agent.id_, ' is: %f', timeAverage)
    # print('Generation time controlers, CBF closed:')
    # for agent in agentsQ:
    #     timeAverage = sum(agent.control_times_)/len(agent.control_times_)
    #     print('Agent ', agent.id_, ' is: %f', timeAverage)

    # print_object_info(leader)
    # plot_.initial_configuration(leader, agents, obstacles)
    # plot_.final_configuration(leader, agents, obstacles)
    # plot_.variables_vs_time_1st(leader, agents, time_data, simTime)
    plot_.variables_vs_time_1st(leaderQ, agentsQ, time_data, simTime)
    # plot_.barrier_action(agents, time_data, simTime)
    # plot_.path(leader, agents, obst=obstacles)
    plot_.path(leaderQ, agentsQ, obst=obstacles)
    # plot_.observed_variables_1st(leader, agents, time_data, simTime)
    
# main()