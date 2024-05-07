'''
    Configuration functions to initialize simulations
'''

import numpy as np

from agent import Agent
from obstacles import Obstacle

def generate_graph(n=3, random=False):
    
    '''
        Generate the connectivity square matrix it must be connected:
            n -> number of agents, by default is 3
            random -> if the graph is random or not, by default is not and it is considered 
                      a inmediate neighbor network
    '''
    
    if random:
        pass
    
    else:
        
        L = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                
                if i == j:
                    
                    L[i][j] = 1
                    
                    if i == n-1:
                        L[i][0] = -1
                    
                elif j == i+1:
                    L[i][j] = -1
        # L = np.array([[3, -1, -1, -1, 0, 0],
        #               [-1, 3, 0, 0, -1, -1],
        #               [-1, 0, 2, -1, 0, 0],
        #               [-1, 0, -1, 2, 0, 0],
        #               [0, -1, 0, 0, 2, -1],
        #               [0, -1, 0, 0, -1, 2]])
    
    return L

def desired_references(agents, leader, distance=4.0, shape=0):
    
    '''
        Set the desired realtive references between agents according differente shapes rotation fixed
            shape    -> 0 - Circle around leader
                        1 - Triangular, leader in front
                        2 - 
            distance -> set the distance between each agent, this must be in the constraints
    '''
    
    if len(agents) >= 2:
        
        if shape == 0:
            # Circular formation
            
            alpha = 2*np.pi / len(agents) # Angular displacement
            r = (distance/2) / np.sin(alpha)
            # constAngle = (np.pi - alpha) + (alpha/2)
            # n = 1 # Number of agent
            for agent in agents:
                
                aux = []
                for i in agent.neighbors_:
                    aux.append(agents[i-1])
                agent.neighbors_ = aux
                agent.disired_distance_ = distance
                agent.generate_displacement_variables(leader)
                
            #     if agent.id_ == 1:
            #         # Considere that the agent id:1 is connected with the leader
                    
            #         agent.l_displacement_x_ = r*np.cos(alpha)
            #         agent.l_displacement_y_ = r*np.sin(alpha)
                
            #     # Displacement angle 
            #     theta = constAngle + n*alpha
            #     if theta >= 2*np.pi:
            #         theta -= 2*np.pi

            #     # Adding the reference with respect to the neighbors
            #     for neighbor in agent.displacement_:

            #         x = distance*np.cos(theta)
            #         y = distance*np.sin(theta)
                    
            #         agent.displacement_[neighbor] = [x, y]
                    
            #     n += 1
                
                # Consider around the leader
                agent.desired_position_[0] = r*np.cos(agent.id_ * alpha)
                agent.desired_position_[1] = r*np.sin(agent.id_ * alpha)
                agent.distance_between_agents_ = distance
                
    else: # Till this moment we suppose that the one agent reachs the consensus with the leader
        
        agents[0].generate_displacement_variables(leader)
        agents[0].distance_between_agents_ = 0.0

def initial_conditions(agent, center, neighbors, random, maxD=10.0, minD=5.0):
    
    '''
        Generate the connectivity square matrix it must be connected:
            agents -> List with the agents (object) to modify their initial conditions
            random -> if the initial positions will be randomly generated
            center -> position of the center of the sample
            maxD   -> maximun distance from the center 
            minD   -> minimum distance from the center 
    '''
    
    agent.neighbors_ = neighbors
    
    if random:
    
        r = np.random.uniform(minD, maxD)
        angle = np.random.uniform(-np.pi, np.pi)
        
        x = center[0] + r*np.cos(angle)
        y = center[1] + r*np.sin(angle)
        
    else: 
        
        r = np.random.uniform(minD, maxD)
        # r = np.random.uniform(1.0, 2.0)
        angle = np.random.uniform(-np.pi, np.pi)
        
        x = -center[0] + r*np.cos(angle)
        y = -center[1] + r*np.sin(angle)
    
    agent.x_ = x
    agent.y_ = y
    
def generate_agents(n, L, obstacles, random=True, lPosition=[0, 0], CBFmethod=3):
    '''
        Generate the agents (object) and initiliza their parameters
            n         -> number of agents
            random    -> if the initial positions will be randomly generated
            lPosition -> initial position of the leader
    '''
    
    leader = Agent(leader=True, pos=lPosition, vel=[0.0, 0.3], acc=[0.0, 0.0], obs=obstacles)
    
    agents = []
    for i in range(n):
        
        agent = Agent(id=i+1, obs=obstacles, CBF=CBFmethod)
        
        if n >= 2:
            
            neighbors = []
            for n, _ in enumerate(L[i]):
                if L[i][n] != 0 and i != n:
                    neighbors.append(int(n+1))
            initial_conditions(agent, lPosition, neighbors, random)
            
        else:
            initial_conditions(agent, lPosition, None, random=False)
        
        agents.append(agent)
        del agent

    desired_references(agents, leader) # Setting the desired formation
    
    return leader, agents

def generate_obstacles(n, randomConfiguration, minR=1.5, maxR=3.0, upperBound=10, lowerBound=-10):
    
    '''
        Generate convex fixed obstacles, with a randomly or non-randomly distribution.
            agents              -> agents that gonna save the obstacles configuration
            n                   -> number of the obstacles
            randomConfiguration -> bool value to generate randomly or not the configuration of the obstacles
            minR                -> minimum radius value
            maxR                -> maximum radius value
            
        **** Note: In this algorithm we suppose that the estimation of the obstacles is solved **** 
    '''
    
    obstacles = []
    for i in range(n):
        obstacles.append(Obstacle(i+1, obstacles, minR, maxR, randomConfiguration, upperBound, lowerBound))
        
    return obstacles