'''
    Here we can plot in time, x-y coordinates and animation simulation. Just 2-D at the moment.
        - Plot initial configuration
        - Plot final configuration
        - Plot paths
        - Plot states vs time
        - Plor errors vs time
        - Animation
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from agent import BFC_data, BFC_data_mod

def initial_configuration(leader, agents, obst=None):
    
    '''
        Here we visualize the initial configuration of the agents and the leader X-Y coordinates
            leader -> the leader object
            agents -> the agents list of objects
            obst   -> if there are obstacles they will be plotted too.
    '''
    
    colors = cmx.rainbow(np.linspace(0, 1, len(agents)))
    
    _, ax = plt.subplots()
    ax.plot(leader.data_pos_x_[0], leader.data_pos_y_[0], '*', color='black', label=r"Leader")
    for i, agent in enumerate(agents):
        ax.plot(agent.data_pos_x_[0], agent.data_pos_y_[0], 'o', color=colors[i], label="Agent {}".format(i+1))
        
        
    if obst:
        
        cirlces = []
        for obs in obst:
            cirlces.append(plt.Circle((obs.x_, obs.y_), obs.radius_, color='black'))
            
        for circle in cirlces:
            ax.add_patch(circle)
            
    ax.set_xlim(-25, 15)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal', adjustable='box')
    plt.title("Initial configuration X-Y coordinates")
    plt.xlabel(r"$x$-coordinate")
    plt.ylabel(r"$y$-coordinate")
    plt.grid()
    plt.legend()
    plt.show()
    
def final_configuration(leader, agents, obst=None):
    
    '''
        Here we visualize the initial configuration of the agents and the leader X-Y coordinates
            leader -> the leader object
            agents -> the agents list of objects
            obst   -> if there are obstacles they will be plotted too.
    '''
    
    colors = cmx.rainbow(np.linspace(0, 1, len(agents)))
    
    _, ax = plt.subplots()
    ax.plot(leader.data_pos_x_[-1], leader.data_pos_y_[-1], '*', color='black', label=r"Leader")
    for i, agent in enumerate(agents):
        ax.plot(agent.data_pos_x_[-1], agent.data_pos_y_[-1], 'o', color=colors[i], label="Agent {}".format(i+1))
        
    if obst:
        
        cirlces = []
        for obs in obst:
            cirlces.append(plt.Circle((obs.x_, obs.y_), obs.radius_, color='black'))
            
        for circle in cirlces:
            ax.add_patch(circle)
            
    ax.set_xlim(-25, 15)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal', adjustable='box')
    plt.title("Final configuration X-Y coordinates")
    plt.xlabel(r"$x$-coordinate")
    plt.ylabel(r"$y$-coordinate")
    plt.grid()
    plt.legend()
    plt.show()
    
def variables_vs_time_1st(leader, agents, t_data, simTime):
    
    '''
        Here we visualize the agents' variables along the time in 1st order
            leader -> the leader object
            agents -> the agents list of objects
            t_data -> time along the simulation
    '''
    
    colors = cmx.rainbow(np.linspace(0, 1, len(agents)))
    
    # Position
    fig, ax = plt.subplots(2, 1)
    fig.suptitle('Positions')
    ax[0].plot(t_data, leader.data_pos_x_, '--', color='black', label=r"Leader")
    for i, agent in enumerate(agents):
        ax[0].plot(t_data, agent.data_pos_x_, '-', color=colors[i], label="Agent {}".format(i+1))
    ax[1].plot(t_data, leader.data_pos_y_, '--', color='black', label=r"Leader")
    for i, agent in enumerate(agents):
        ax[1].plot(t_data, agent.data_pos_y_, '-', color=colors[i], label="Agent {}".format(i+1))   
    ax[0].set_ylabel(r"$x$-coord")
    ax[1].set_ylabel(r"$y$-coord")
    ax[0].set_xlim(0, simTime)
    ax[1].set_xlim(0, simTime)
    ax[0].grid()
    ax[1].grid()
    plt.xlabel(r"Time $(s)$")
    plt.legend()
    
    # Velocity
    figx, axx = plt.subplots(2, 1)
    figx.suptitle('Velocities')
    axx[0].plot(t_data, leader.data_vel_x_, '--', color='black', label=r"Leader")
    for i, agent in enumerate(agents):
        axx[0].plot(t_data, agent.data_vel_x_, '-', color=colors[i], label="Agent {}".format(i+1))
    axx[1].plot(t_data, leader.data_vel_y_, '--', color='black', label=r"Leader")
    for i, agent in enumerate(agents):
        axx[1].plot(t_data, agent.data_vel_y_, '-', color=colors[i], label="Agent {}".format(i+1))   
    axx[0].set_ylabel(r"$x$-vel")
    axx[1].set_ylabel(r"$y$-vel")
    axx[0].set_xlim(0, simTime)
    axx[1].set_xlim(0, simTime)
    axx[0].grid()
    axx[1].grid()
    plt.xlabel(r"Time $(s)$")
    plt.legend()
    
    # Desired Velocity
    figd, axd = plt.subplots(2, 1)
    figd.suptitle('Desired Velocities')
    for i, agent in enumerate(agents):
        axd[0].plot(t_data, agent.data_desired_vel_x_, '-', color=colors[i], label="Agent {}".format(i+1))
        axd[1].plot(t_data, agent.data_desired_vel_y_, '-', color=colors[i], label="Agent {}".format(i+1))   
    axd[0].set_ylabel(r"$x$-vel")
    axd[1].set_ylabel(r"$y$-vel")
    axd[0].set_xlim(0, simTime)
    axd[1].set_xlim(0, simTime)
    axd[0].grid()
    axd[1].grid()
    plt.xlabel(r"Time $(s)$")
    plt.legend()
    
    # Safety Velocity
    figs, axs = plt.subplots(2, 1)
    figs.suptitle('Safety Velocities')
    for i, agent in enumerate(agents):
        axs[0].plot(t_data, agent.data_safety_vel_x_, '-', color=colors[i], label="Agent {}".format(i+1))
        axs[1].plot(t_data, agent.data_safety_vel_y_, '-', color=colors[i], label="Agent {}".format(i+1))   
    axs[0].set_ylabel(r"$x$-vel")
    axs[1].set_ylabel(r"$y$-vel")
    axs[0].set_xlim(0, simTime)
    axs[1].set_xlim(0, simTime)
    axs[0].grid()
    axs[1].grid()
    plt.xlabel(r"Time $(s)$")
    plt.legend()
    
    # Distances between neighbors and obstacles
    if agents[0].neighbors_:
        ref = agents[0].radius_*1.2
        collided = agents[0].radius_
        agNum = len(agents)
        figr, axr = plt.subplots(agNum, 1, sharex=True)
        figr.suptitle('Distances between agents')
        for i, agent in enumerate(agents):
            axr[i].hlines(y=ref, xmin=0.0, xmax=simTime, color='black', linestyle='dashed', label="Agents reference")
            axr[i].hlines(y=collided, xmin=0.0, xmax=simTime, color='red', linestyle='dashed', label="Agents collided")
            for idx in agent.data_neighbors_distances_:
                axr[i].plot(t_data, agent.data_neighbors_distances_[idx], '-', color=colors[idx-1], label="Agent {}".format(idx))      
            axr[i].set_ylabel(r"Distances")
            axr[i].grid()
        plt.xlabel(r"Time $(s)$")
        plt.xlim(0, simTime)
        plt.legend()
    
    # Distance with the obstacles
    if agents[0].obstacles_:
        if len(agents[0].obstacles_) >= 2:
            obstacles = agents[0].obstacles_
            obsNum = len(obstacles)
            figo, axo = plt.subplots(obsNum, 1, sharex=True)
            figo.suptitle('Agents - Obstacles distances')
            for i, obs in enumerate(obstacles):
                axo[i].hlines(y=obs.radius_*1.2, xmin=0.0, xmax=simTime, color='black', linestyle='dashed', label="Obstacle {} reference".format(obs.tag_))
                axo[i].hlines(y=obs.radius_, xmin=0.0, xmax=simTime, color='red', linestyle='dashed', label="Obstacle {} collided".format(obs.tag_))
                for j, agent in enumerate(agents):
                    axo[i].plot(t_data, agent.data_obstacles_distances_[obs.tag_], '-', color=colors[j], label="Agent {}".format(agent.id_))
                axo[i].set_ylabel(r"Distances")
                axo[i].grid()
            plt.xlabel(r"Time $(s)$")
            plt.xlim(0, simTime)
            plt.legend()
    
    fig_psi, ax_psi = plt.subplots(len(agents), 1, sharex=True)
    fig_psi.suptitle('Psi functions')
    for i, agent in enumerate(agents):
        
        for psi in agent.psi_0:
            try:
                ax_psi[i].plot(t_data, agent.psi_0[psi], '-', color=colors[i])
                ax_psi[i].plot(t_data, agent.psi_1[psi], '-', color=colors[i])
            except:
                print(agent.psi_0[psi])
                print(agent.psi_1[psi])
            
        ax_psi[i].set_ylabel(r"Psi value")
        ax_psi[i].grid()
        
    plt.xlabel(r"Time $(s)$")
    plt.xlim(0, simTime)
    
    # Show the plots
    plt.show()

def path(leader, agents, obst=None):
    
    '''
        Here we visualize the path of the agents and the leader X-Y coordinates
            leader -> the leader object
            agents -> the agents list of objects
            obst   -> if there are obstacles they will be plotted too.
    '''
    
    colors = cmx.rainbow(np.linspace(0, 1, len(agents)))
    
    _, ax = plt.subplots()
    ax.plot(leader.data_pos_x_, leader.data_pos_y_, '--', color='black', label=r"Leader")
    ax.plot(leader.data_pos_x_[-1], leader.data_pos_y_[-1], '*', color='black')
    for i, agent in enumerate(agents):
        ax.plot(agent.data_pos_x_, agent.data_pos_y_, '-', color=colors[i], label="Agent {}".format(i+1))
        ax.plot(agent.data_pos_x_[-1], agent.data_pos_y_[-1], 'o', color=colors[i])
        for c in range(1, 100):
            alpha = 1/c
            robot = plt.Circle((agent.data_pos_x_[int(len(agent.data_pos_x_)/(10*c))], 
                                agent.data_pos_y_[int(len(agent.data_pos_x_)/(10*c))]), 
                                agent.radius_, 
                                color=colors[i],
                                alpha=alpha)
            ax.add_patch(robot)
        
    if obst:
        
        cirlces = []
        for obs in obst:
            cirlces.append(plt.Circle((obs.x_, obs.y_), obs.radius_, color='black'))
            
        for circle in cirlces:
            ax.add_patch(circle)
    
    ax.set_xlim(-25, 15)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal', adjustable='box')
    plt.title("Trajectories X-Y coordinates")
    plt.xlabel(r"$x$-coordinate")
    plt.ylabel(r"$y$-coordinate")
    plt.grid()
    plt.legend()
    plt.show()
    
def observed_variables_1st(leader, agents, t_data, simTime):
    
    '''
        Here we visualize the observer error of the states in 1st order
            leader -> the leader object
            agents -> the agents list of objects
    '''
    
    colors = cmx.rainbow(np.linspace(0, 1, len(agents)))
    
    # Position
    fig, ax = plt.subplots(2, 1)
    fig.suptitle('Observer Position estimation')
    ax[0].plot(t_data, leader.data_pos_x_, '--', color='black', label=r"Leader")
    for i, agent in enumerate(agents):
        ax[0].plot(t_data, agent.data_l_pos_x_, '-', color=colors[i], label="Agent {}".format(i+1))
    ax[1].plot(t_data, leader.data_pos_y_, '--', color='black', label=r"Leader")
    for i, agent in enumerate(agents):
        ax[1].plot(t_data, agent.data_l_pos_y_, '-', color=colors[i], label="Agent {}".format(i+1))   
    ax[0].set_ylabel(r"$x$-coord")
    ax[1].set_ylabel(r"$y$-coord")
    ax[0].set_xlim(0, simTime)
    ax[1].set_xlim(0, simTime)
    ax[0].grid()
    ax[1].grid()
    plt.xlabel(r"Time $(s)$")
    plt.legend()
    
    # Velocity
    figx, axx = plt.subplots(2, 1)
    figx.suptitle('Observer Velocity estimation')
    axx[0].plot(t_data, leader.data_vel_x_, '--', color='black', label=r"Leader")
    for i, agent in enumerate(agents):
        axx[0].plot(t_data, agent.data_l_vel_x_, '-', color=colors[i], label="Agent {}".format(i+1))
    axx[1].plot(t_data, leader.data_vel_y_, '--', color='black', label=r"Leader")
    for i, agent in enumerate(agents):
        axx[1].plot(t_data, agent.data_l_vel_y_, '-', color=colors[i], label="Agent {}".format(i+1))   
    axx[0].set_ylabel(r"$x$-vel")
    axx[1].set_ylabel(r"$y$-vel")
    axx[0].set_xlim(0, simTime)
    axx[1].set_xlim(0, simTime)
    axx[0].grid()
    axx[1].grid()
    plt.xlabel(r"Time $(s)$")
    plt.legend()
    plt.show()
    
def barrier_action(agents, t_data, simTime):
    
    '''
        Plot both the formation control sates and barrier function deformations
    '''
    
    colors = cmx.rainbow(np.linspace(0, 1, len(agents)))
    
    # # Position
    # fig, ax = plt.subplots(2, 1)
    # fig.suptitle('Only formation positions')
    # for i, agent in enumerate(agents):
    #     ax[0].plot(t_data, agent.f_data_x_, '-', color=colors[i], label="Agent {}".format(i+1))
    #     ax[1].plot(t_data, agent.f_data_x_, '-', color=colors[i], label="Agent {}".format(i+1))   
    # ax[0].set_ylabel(r"$x$-coord")
    # ax[1].set_ylabel(r"$y$-coord")
    # ax[0].set_xlim(0, simTime)
    # ax[1].set_xlim(0, simTime)
    # ax[0].grid()
    # ax[1].grid()
    # plt.xlabel(r"Time $(s)$")
    # plt.legend()
    
    # # Barrier function positions modifications
    # figB, axB = plt.subplots(2, 1)
    # figB.suptitle('Barrier function action in positions')
    # for i, agent in enumerate(agents):
    #     axB[0].plot(t_data, agent.w_data_x_, '-', color=colors[i], label="Agent {}".format(i+1))
    #     axB[1].plot(t_data, agent.w_data_x_, '-', color=colors[i], label="Agent {}".format(i+1))   
    # axB[0].set_ylabel(r"$x$-coord")
    # axB[1].set_ylabel(r"$y$-coord")
    # axB[0].set_xlim(0, simTime)
    # axB[1].set_xlim(0, simTime)
    # axB[0].grid()
    # axB[1].grid()
    # plt.xlabel(r"Time $(s)$")
    # plt.legend()
    
    # # Velocity
    # figx, axx = plt.subplots(2, 1)
    # figx.suptitle('Only formation velocities')
    # for i, agent in enumerate(agents):
    #     axx[0].plot(t_data, agent.f_data_vel_x_, '-', color=colors[i], label="Agent {}".format(i+1))
    #     axx[1].plot(t_data, agent.f_data_vel_x_, '-', color=colors[i], label="Agent {}".format(i+1))   
    # axx[0].set_ylabel(r"$x$-vel")
    # axx[1].set_ylabel(r"$y$-vel")
    # axx[0].set_xlim(0, simTime)
    # axx[1].set_xlim(0, simTime)
    # axx[0].grid()
    # axx[1].grid()
    # plt.xlabel(r"Time $(s)$")
    # plt.legend()
    
    # # Barrier function velicities modifications
    # figxB, axxB = plt.subplots(2, 1)
    # figxB.suptitle('Barrier function action in velocities')
    # for i, agent in enumerate(agents):
    #     axxB[0].plot(t_data, agent.w_data_vel_x_, '-', color=colors[i], label="Agent {}".format(i+1))
    #     axxB[1].plot(t_data, agent.w_data_vel_x_, '-', color=colors[i], label="Agent {}".format(i+1))   
    # axxB[0].set_ylabel(r"$x$-vel")
    # axxB[1].set_ylabel(r"$y$-vel")
    # axxB[0].set_xlim(0, simTime)
    # axxB[1].set_xlim(0, simTime)
    # axxB[0].grid()
    # axxB[1].grid()
    # plt.xlabel(r"Time $(s)$")
    # plt.legend()
    
    # Plot the barrier function #######################################################################
    zx = np.arange(-4.1, 4.1, 0.01)
    zy = np.arange(-4.1, 4.1, 0.01)
    zX, zY = np.meshgrid(zx, zy)
    zs = np.array(BFC_data(np.ravel(zX), np.ravel(zY), 2))
    zmod = np.array(BFC_data_mod(np.ravel(zX), np.ravel(zY), 2))
    Z = zs.reshape(zX.shape)
    Zmod = zmod.reshape(zX.shape)
    
    # Gx, Gy = np.gradient(Zmod)
    # gx, gy = np.array(BFC_data_gradient(zX, zY))
    # Gx = gx.reshape(zX.shape)
    # Gy = gy.reshape(zX.shape)
    # G = (Gx**2 + Gy**2)**0.5
    # N = G/G.max()
    
    # Surface of the barrier function
    fig3, ax3 = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax3.plot_surface(zX, zY, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig3.colorbar(surf, shrink=0.5, aspect=5)
    
    fig4, ax4 = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax4.plot_surface(zX, zY, Zmod, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig4.colorbar(surf, shrink=0.5, aspect=5)
    
    # fig5, ax5 = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax5.plot_surface(zX, zY, Gx, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # fig5.colorbar(surf, shrink=0.5, aspect=5)
    
    # fig6, ax6 = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax6.plot_surface(zX, zY, Gy, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # fig6.colorbar(surf, shrink=0.5, aspect=5)
    ###################################################################################################
    
    plt.show()