# Agent class, here we save information and compute the local controller

import numpy as np
import osqp
from time import time
from scipy.special import gamma
from scipy import sparse
from shapely.geometry import Point

class Agent:

    def __init__(self, id=None, leader=False, pos=[0,0], vel=[0,0], acc=[0,0], r=0.6, obs=None, CBF=1):
        
        # Local variables
        self.leader_role_ = leader
        self.id_ = id
        self.neighbors_ = []
        self.radius_ = r 
        self.obstacles_ = obs
        self.x_ = pos[0]
        self.y_ = pos[1]
        self.vel_x_ = vel[0]
        self.vel_y_ = vel[1]
        self.acc_x_ = acc[0]
        self.acc_y_ = acc[1]
        self.acc_max_ = 2.2
        self.CFB_method = CBF
        self.agents_QP_solver_ = None
        self.obstacles_QP_solver_ = None
        
        # Desired control
        self.ud_x_ = self.vel_x_
        self.ud_y_ = self.vel_y_
        # Safety control
        self.us_x_ = 0.0
        self.us_y_ = 0.0
        
        # Saving data: Information of the performance of the agent
        self.data_pos_x_ = []
        self.data_pos_y_ = []
        self.data_vel_x_ = []
        self.data_vel_y_ = []
        self.data_acc_x_ = []
        self.data_acc_y_ = []
        
        self.data_desired_vel_x_ = []
        self.data_desired_vel_y_ = []
        self.data_safety_vel_x_ = []
        self.data_safety_vel_y_ = []
        
        self.psi_0 = {}
        self.psi_1 = {}
        self.psi_ag = []
        self.psi_obs = []
        
        self.data_neighbors_distances_ = {}
        self.data_obstacles_distances_ = {}
        
        self.control_times_ = []
        self.break_ = False
        
    def config_data_distances(self, agents):
        
        '''
            Configure the arrays in the dictionaries
        '''
        
        if self.neighbors_:
            for agent in agents:
                if self.id_ != agent.id_:  
                    self.data_neighbors_distances_[agent.id_] = []
        
        if self.obstacles_: 
            for obs in self.obstacles_:
                self.data_obstacles_distances_[obs.tag_] = []
        
    def leader_dynamic(self, t, dt):
        
        '''
            Leader dynamic or trjectory of the leader
        '''
        
        if not self.leader_role_:
            return
        else:
            
            self.acc_x_ = -0.018*np.cos(0.06*t)
            self.acc_y_ = -0.018*np.sin(0.06*t)
            # self.acc_x_ = 0.0
            # self.acc_y_ = 0.0
            
            self.vel_x_ += self.acc_x_*dt
            self.vel_y_ += self.acc_y_*dt
            
            self.x_ += self.vel_x_*dt
            self.y_ += self.vel_y_*dt   
        
    def leader_fixed(self, t, dt):
        
        '''
            Leader fixed, this function help us to define a fixed goal where we can reach it in formation or consensus
        '''       
        
        if not self.leader_role_:
            return
        
        self.acc_x_ = 0.0
        self.acc_y_ = 0.0
        
        self.vel_x_ = 0.0
        self.vel_y_ = 0.0
        
    def generate_displacement_variables(self, leader=None):
        
        if self.id_ == 1:
            # It has leader comunication
            
            self.l_displacement_x_ = 0.0
            self.l_displacement_y_ = 0.0
            self.b_ = 1 # Comunication with the leader value
            
        else:
            self.b_ = 0
        
        self.leader_ = leader
        
        if self.neighbors_:
            self.displacement_ = {}
            for n in self.neighbors_:
                self.displacement_[n.id_] = [0.0, 0.0]
            
        self.desired_position_ = [0.0, 0.0]
        
    def set_observer_1st(self, method=1):
        
        '''
            This function add observer variables to estimate the leader variables.
            Here is considered as first order observer
                method -> Selet the method to estimate the leader states:
                            1: Observer-Bases Distributed Leader-Follower Tracking Control: A New
                               Perspective and Results / Chuan Yan, and Huazhen Fang
                            2: Observer-Based Leader-Follower Consensus Tracking with Fixed-time 
                               Convergence / M. A. Trujillo, D. Gomez-Gutierrez, M. Defoort,
                               J. Ruiz-Leon, and H. M. Becerra
                            3: Observer / Rodrigo Aldana
        '''
        
        # Initialize the leader estimation states
        self.l_pos_x_ = self.x_
        self.l_pos_y_ = self.y_
        self.l_vel_x_ = 0.0
        self.l_vel_y_ = 0.0
        self.observer_method_ = method
        self.observer_order_ = 1
        
        if method == 1:
            self.z_x_ = 0.0
            self.z_y_ = 0.0
            self.d_x_ = 0.0
            self.d_y_ = 0.0
            
        # Save data
        self.data_l_pos_x_ = [] 
        self.data_l_pos_y_ = [] 
        self.data_l_vel_x_ = [] 
        self.data_l_vel_y_ = [] 
        # and errors
        self.error_pos_x_ = []
        self.error_pos_y_ = []
        self.error_vel_x_ = []
        self.error_vel_y_ = []
        
    def set_observer_2nd(self, method=1):
        
        '''
            This function add observer variables to estimate the leader variables.
            Here is considered as first order observer
                method -> Selet the method to estimate the leader states:
                            1: Observer-Bases Distributed Leader-Follower Tracking Control: A New
                               Perspective and Results / Chuan Yan, and Huazhen Fang
                            2: Observer-Based Leader-Follower Consensus Tracking with Fixed-time 
                               Convergence / M. A. Trujillo, D. Gomez-Gutierrez, M. Defoort,
                               J. Ruiz-Leon, and H. M. Becerra
                            3: Observer / Rodrigo Aldana
        '''
        
        # Initialize the leader estimation states
        self.l_pos_x_ = self.x_
        self.l_pos_y_ = self.y_
        self.l_vel_x_ = 0.0
        self.l_vel_y_ = 0.0
        self.l_acc_x_ = 0.0
        self.l_acc_y_ = 0.0
        self.observer_method_ = method
        self.observer_order_ = 2
        
        if method == 1:
            
            # Observer internal variables
            self.z_x_ = 0.0
            self.z_y_ = 0.0
            self.d_x_ = 0.0
            self.d_y_ = 0.0
            
        elif method == 2:
            
            # Observer parameters
            self.alpha_ = 0.4
            self.beta_ = 0.6
            self.p_ = 0.2
            self.q_ = 1.5
            self.k_ = 1.0
            g_1 = gamma((1 - self.k_*self.p_)/(self.q_ - self.p_)) # Gamma functions
            g_2 = gamma((self.k_*self.q_ - 1)/(self.q_ - self.p_))
            g_3 = gamma(self.k_)
            self.ro_ = ((g_1 * g_2)/((self.alpha_**self.k_)*g_3*(self.q_ - self.p_)))*((self.alpha_/self.beta_)**((1 - self.k_*self.p_)/(self.q_ - self.p_)))
            
        elif method == 3: 
            
            # Observer parameters
            m = 2 # Number of derivatives 
            alphas = []
            for mu in range(m+1):
                alphas.append((m - mu)/(m+1))
            self.alphas_ = alphas
        
        # Save data
        self.data_l_pos_x_ = [] 
        self.data_l_pos_y_ = [] 
        self.data_l_vel_x_ = [] 
        self.data_l_vel_y_ = [] 
        self.data_l_acc_x_ = [] 
        self.data_l_acc_y_ = [] 
        # and errors
        self.error_pos_x_ = []
        self.error_pos_y_ = []
        self.error_vel_x_ = []
        self.error_vel_y_ = []
        self.error_acc_x_ = []
        self.error_acc_y_ = []
        
    def observation(self, dt, l=0.01, tau=1.15, c=10.5):
        
        '''
            Here a sample of the observer y performed according to the method and order selected
                neighbors -> neighbors with the agent has comunication
                dt        -> sample time or integration time
                l         -> scalar gain
                tau       -> scalar gain
                c         -> scalar gain
        '''
        
        b = self.b_
        # b = 1
        z_x = self.z_x_
        z_y = self.z_y_
        d_x = self.d_x_
        d_y = self.d_y_
        
        if self.observer_method_ == 1:
            
            if self.observer_order_ == 1:
                
                aux = [0.0, 0.0]
                aux_1 = [0.0, 0.0]
                for n in self.neighbors_:
                    
                    aux[0] += (self.l_pos_x_ - n.l_pos_x_) 
                    aux[1] += (self.l_pos_y_ - n.l_pos_y_)
                    
                    aux_1[0] += (self.l_vel_x_ - n.l_vel_x_)
                    aux_1[1] += (self.l_vel_y_ - n.l_vel_y_)
                    print(aux_1)
                
                # z_dot_x = - b*l*z_x - (b)*(l**2)*self.leader_.x_ - aux_1[0] - d_x*np.sign(aux_1[0] + l*b*(self.l_vel_x_ - self.leader_.vel_x_))
                # z_dot_y = - b*l*z_y - (b)*(l**2)*self.leader_.y_ - aux_1[1] - d_y*np.sign(aux_1[1] + l*b*(self.l_vel_y_ - self.leader_.vel_y_))
                z_dot_x = - l*(aux_1[0] + b*(self.l_vel_x_ - self.leader_.vel_x_))
                z_dot_y = - l*(aux_1[1] + b*(self.l_vel_y_ - self.leader_.vel_y_))
                z_x += z_dot_x * dt
                z_y += z_dot_y * dt
                
                if self.id_ == 1:
                    print(l*b*(self.l_vel_x_ - self.leader_.vel_x_))
                    print(z_x)
                    print(z_y)
                
                d_dot_x = tau*np.abs(aux_1[0] + l*b*(self.l_vel_x_ - self.leader_.vel_x_))
                d_dot_y = tau*np.abs(aux_1[1] + l*b*(self.l_vel_y_ - self.leader_.vel_y_))
                d_x += d_dot_x * dt
                d_y += d_dot_y * dt
                
                # Velocity observed
                u_x = z_x #+ b*l*self.leader_.x_
                u_y = z_y #+ b*l*self.leader_.y_
                # u_x = -l*(aux_1[0] + b*(self.l_vel_x_ - self.leader_.vel_x_))
                # u_y = -l*(aux_1[1] + b*(self.l_vel_y_ - self.leader_.vel_y_))
                
                # Position estimation
                p_dot_x = -c*(aux[0] + b*(self.l_pos_x_ - self.leader_.x_)) + u_x
                p_dot_y = -c*(aux[1] + b*(self.l_pos_y_ - self.leader_.y_)) + u_y
                
                # Update states
                self.l_vel_x_ = u_x
                self.l_vel_y_ = u_y
                self.l_pos_x_ += p_dot_x*dt
                self.l_pos_y_ += p_dot_y*dt
                self.d_x_ = d_x
                self.d_x_ = d_y
                self.z_x_ = z_x 
                self.z_x_ = z_y 
                
            elif self.observer_order_ == 2:
                
                sum_vel = [0.0, 0.0]
                sum_pos = [0.0, 0.0]
                for n in self.neighbors_:
                    
                    sum_vel[0] += self.l_vel_x_ - n.l_vel_x_
                    sum_vel[1] += self.l_vel_y_ - n.l_vel_y_
                    
                    sum_pos[0] += self.l_pos_x_ - n.l_pos_x_
                    sum_pos[1] += self.l_pos_y_ - n.l_pos_y_
                    
                vel_error_x = self.l_vel_x_ - self.leader_.vel_x_
                vel_error_y = self.l_vel_y_ - self.leader_.vel_y_
                
                pos_error_x = self.l_pos_x_ - self.leader_.x_
                pos_error_y = self.l_pos_y_ - self.leader_.y_
                    
                u_x = - sum_vel[0] - b*vel_error_x - d_x*np.sign(sum_vel[0] + b*vel_error_x)
                u_y = - sum_vel[1] - b*vel_error_y - d_y*np.sign(sum_vel[1] + b*vel_error_y)
                
                d_dot_x = tau*np.abs(sum_vel[0] + b*vel_error_x)
                d_dot_y = tau*np.abs(sum_vel[1] + b*vel_error_y)
                d_x += d_dot_x*dt
                d_y += d_dot_y*dt
                
                z_dot_x = -b*l*z_x - b*(l**2)*self.leader_.x_ - sum_vel[0] + u_x*dt
                z_dot_y = -b*l*z_y - b*(l**2)*self.leader_.y_ - sum_vel[1] + u_y*dt
                z_x += z_dot_x*dt
                z_y += z_dot_y*dt
                
                vel_x = z_x + b*l*self.leader_.x_
                vel_y = z_y + b*l*self.leader_.y_
                
                x_dot = -c*(sum_pos[0] + b*pos_error_x) + vel_x
                y_dot = -c*(sum_pos[1] + b*pos_error_y) + vel_y
                
                # Update states
                self.l_acc_x_ = u_x
                self.l_acc_y_ = u_y
                self.l_vel_x_ = x_dot
                self.l_vel_y_ = y_dot
                self.l_pos_x_ += x_dot*dt
                self.l_pos_y_ += y_dot*dt
                self.d_x_ = d_x
                self.d_x_ = d_y
                self.z_x_ = z_x 
                self.z_x_ = z_y 
                
        elif self.observer_method_ == 2:
            
            if self.observer_order_ == 2:
                
                def Gamma(t, z):
                    return np.exp(-t) * t**(z-1)
                
    def formation_control(self, dt, c1=0.9, BFC=False):
        
        '''
            A formation control technic is perfoming to reach a predefined formation, it could be:
                - Displacement based a leader, as centroid of the formation
                - Relative displacement with agent's neighbors.
            
            Variables
                dt  -> Sample time
                c1  -> position gain
                BFC -> define if the Barrier Function Control is activated
                
            --------------Improve it to high order!!!!---------------------
        '''
        
        # Variables to get the time that the control spend
        tic = time()
        
        
        # First order formation control
        desiredX = self.leader_.x_ + self.desired_position_[0]
        desiredY = self.leader_.y_ + self.desired_position_[1]

        errorX = self.x_ - desiredX
        errorY = self.y_ - desiredY
        
        if BFC:
            
            self.ud_x_ = -c1*errorX + self.leader_.vel_x_
            self.ud_y_ = -c1*errorY + self.leader_.vel_y_
            
            bfcX, bfcY = self.BFC_action()
            ux = bfcX
            uy = bfcY
            
            self.us_x_ = ux - self.ud_x_
            self.us_y_ = uy - self.ud_y_
        
        else:
        
            ux = -c1*errorX + self.leader_.vel_x_
            uy = -c1*errorY + self.leader_.vel_y_
            
        toc = time() - tic
            
        x = self.x_ + ux*dt
        y = self.y_ + uy*dt
        
        self.x_ = x 
        self.y_ = y
        self.vel_x_ = ux 
        self.vel_y_ = uy 
        self.control_times_.append(toc)
        
    def formation_control_HO(self, dt, c1=10.5, c2=7.5):
        
        '''
            A formation control technic is perfoming to reach a predefined formation, it could be:
                - Displacement based a leader, as centroid of the formation
                - Relative displacement with agent's neighbors.
            
            Variables
                dt  -> Sample time
                c1  -> position gain
        '''
        
        # Variables to get the time that the control spend
        tic = time()
        
        # First order formation control
        desiredX = self.leader_.x_ + self.desired_position_[0]
        desiredY = self.leader_.y_ + self.desired_position_[1]

        pos_errorX = self.x_ - desiredX
        pos_errorY = self.y_ - desiredY
        vel_errorX = self.vel_x_ - self.leader_.vel_x_
        vel_errorY = self.vel_y_ - self.leader_.vel_y_
             
        self.ud_x_ = -c1*pos_errorX - c2*vel_errorX + self.leader_.acc_x_
        self.ud_y_ = -c1*pos_errorY - c2*vel_errorY + self.leader_.acc_y_
            
        bfcX, bfcY = self.HOCBF_action()
        ux = bfcX
        uy = bfcY
        # ux = self.ud_x_
        # uy = self.ud_y_
            
        self.us_x_ = ux - self.ud_x_
        self.us_y_ = uy - self.ud_y_
            
        toc = time() - tic
            
        vx = self.vel_x_ + ux*dt
        vy = self.vel_y_ + uy*dt
        x = self.x_ + vx*dt
        y = self.y_ + vy*dt
        
        self.x_ = x 
        self.y_ = y
        self.vel_x_ = vx 
        self.vel_y_ = vy 
        self.acc_x_ = ux 
        self.acc_y_ = uy 
        self.control_times_.append(toc)
    
    def BFC_action(self, delta_sup=5000.0, alpha_obs=0.2, alpha_neigh=0.4):
        
        '''
            A Barrier Function Control is developed to avoid collisions
                error     -> diference between the agent and its desired position
                delta_sup -> maximum allowed distance
                delta_inf -> minimum allowed distance
                
            New data for the CBF extension with alpha function and QP
                cbf_alpha
        '''
        
        method = self.CFB_method
            
        if method == 1: # Weigh recentered barrier function scheme
                        
            # Control gains
            c1 = 0.5
            
            Bx = 0.0         
            By = 0.0  
            ex = 0.0
            ey = 0.0       
                
            # Agents
            if self.neighbors_:
                
                # Constraints parameters
                delta_inf = self.radius_*1.2
            
                # norm1 = np.linalg.norm(np.array(self.desired_position_))
                norm1 = self.disired_distance_
                k1 = (delta_inf**2)/( (norm1**2)*(norm1**2 - delta_inf**2) )
                k2 = 1/(delta_sup**2 - norm1**2)
                
                for n in self.neighbors_:
                    
                    zx = self.x_ - n.x_
                    zy = self.y_ - n.y_
                    
                    ex += zx - self.disired_distance_
                    ey += zy - self.disired_distance_
                    
                    Bx += k1*( np.abs(zx)/((delta_sup**2) - (zx**2) - (zy**2)) ) + k2*( (-(delta_inf**2)*np.abs(zx)) /
                                                        (((zx**2) + (zy**2))*(-(delta_inf**2) + (zx**2) + (zy**2))) )
                    By += k1*( np.abs(zy)/((delta_sup**2) - (zx**2) - (zy**2)) ) + k2*( (-(delta_inf**2)*np.abs(zy)) /
                                                        (((zx**2) + (zy**2))*(-(delta_inf**2) + (zx**2) + (zy**2))) )
            
            # Obstacles
            if self.obstacles_:

                for obs in self.obstacles_:
                    
                    norm = np.sqrt((self.leader_.x_ - obs.x_)**2 + (self.leader_.y_ - obs.y_)**2)
                    k = (obs.radius_**2)/( (norm**2)*(norm**2 - obs.radius_**2) )
                    # k = 0.001
                    
                    zx_o = self.x_ - obs.x_
                    zy_o = self.y_ - obs.y_
                    
                    ex += zx_o - obs.radius_*1.5
                    ey += zy_o - obs.radius_*1.5

                    Bx += k*( (-((obs.radius_*1.5)**2)*np.abs(zx_o)) / (((zx_o**2) + (zy_o**2))*(-((obs.radius_*1.5)**2) + (zx_o**2) + (zy_o**2))) )
                    By += k*( (-((obs.radius_*1.5)**2)*np.abs(zy_o)) / (((zx_o**2) + (zy_o**2))*(-((obs.radius_*1.5)**2) + (zx_o**2) + (zy_o**2))) )
            
            # Reference errors to reach the consensus
            # ex /= len(self.neighbors_)
            # ey /= len(self.neighbors_)
            
            wGx = Bx
            wGy = By
            
            ux = -c1*wGx + self.ud_x_
            uy = -c1*wGy + self.ud_y_
            
            return ux, uy
        
        elif method == 2: # CBF-QP
            
            udX = self.ud_x_
            udY = self.ud_y_
            
            if self.neighbors_:
                
                # Calculate barrier values and coeffs in h_dot to avoid agents
                self.calculate_h_and_coeffs(self.neighbors_)
                
                # Define the problem data
                self.define_QP_problem_data(udX, udY, alpha_neigh)
                
                if self.agents_QP_solver_ is None or self.new_QP_:
                    
                    # Creat
                    self.agents_QP_solver_ = osqp.OSQP()
                    self.agents_QP_solver_.setup(self.P_, self.q_, self.A_, self.l_, self.u_, verbose=False, time_limit=0)
                
                else:
                    
                    # Update
                    self.agents_QP_solver_.update(q=self.q_, l=self.l_, u=self.u_, Ax=self.A_.data)
                    
                # Solve QP problem
                res = self.agents_QP_solver_.solve()
                udX, udY, _ = res.x 
            
            # For infeasible solution
            udX = self.ud_x_ if udX is None else udX
            udY = self.ud_y_ if udY is None else udY
            
            ux = None  
            uy = None  
            if self.obstacles_:
                
                # Calculate barrier values and coeffs in h_dot to avoid obstacles
                self.calculate_h_and_coeffs(self.obstacles_)
                
                # Define the problem data
                self.define_QP_problem_data(udX, udY, alpha_obs)
                
                if self.obstacles_QP_solver_ is None or self.new_QP_:
                    
                    # Creat
                    self.obstacles_QP_solver_ = osqp.OSQP()
                    self.obstacles_QP_solver_.setup(self.P_, self.q_, self.A_, self.l_, self.u_, verbose=False, time_limit=0)
                
                else:
                    
                    # Update
                    self.obstacles_QP_solver_.update(q=self.q_, l=self.l_, u=self.u_, Ax=self.A_.data)
                    
                # Solve QP problem
                res = self.obstacles_QP_solver_.solve()
                ux, uy, _ = res.x 
            
            # For infeasible solution
            ux = udX if ux is None else ux
            uy = udY if uy is None else uy
            
            return ux, uy
        
        elif method == 3: # Add the neighbors and obstacles in the same QP
            
            udX = self.ud_x_
            udY = self.ud_y_
            
            agents_and_obs = []
            
            if self.obstacles_:
                agents_and_obs.extend(self.obstacles_)
            
            if self.neighbors_:
                agents_and_obs.extend(self.neighbors_)
                
            # Calculate barrier values and coeffs in h_dot to avoid agents
            self.calculate_h_and_coeffs(agents_and_obs)
                
            # Define the problem data
            self.define_QP_problem_data(udX, udY, alpha_obs)
                
            if self.agents_QP_solver_ is None or self.new_QP_:
                    
                # Creat
                self.agents_QP_solver_ = osqp.OSQP()
                self.agents_QP_solver_.setup(self.P_, self.q_, self.A_, self.l_, self.u_, verbose=False, time_limit=0)
                
            else:
                    
                # Update
                self.agents_QP_solver_.update(q=self.q_, l=self.l_, u=self.u_, Ax=self.A_.data)
                    
            # Solve QP problem
            res = self.agents_QP_solver_.solve()
            udX, udY, _ = res.x 
            
            # For infeasible solution
            udX = self.ud_x_ if udX is None else udX
            udY = self.ud_y_ if udY is None else udY
            
            return udX, udY
        
        elif method == 4: # Closed form 
            
            '''
                For this method considere that the control applied in the agent is:
                    u(x, t) = u_d(x, t) + u_s(x, t)
                where u_d is the desired control or objetive control and u_s is the safety control
            '''
            # Calculate safety controls
            
            udX = self.ud_x_
            udY = self.ud_y_
            
            # Avoid agents
            usX = 0.0
            usY = 0.0
            if self.neighbors_:
                
                for n in self.neighbors_:
                    
                    psi = self.psi_function(n, alpha_neigh, udX, udY)
                    
                    # Errors
                    e_x = self.x_ - n.x_
                    e_y = self.y_ - n.y_
                    
                    # Safety controls
                    if psi < 0:
                        usX += -0.5*(e_x/(e_x**2 + e_y**2))*psi
                        usY += -0.5*(e_y/(e_x**2 + e_y**2))*psi
                    else:
                        usX += 0                    
                        usY += 0
                                            
                # Control to avoid just neighbors agents
                udX += usX
                udY += usY
            
            if self.obstacles_:
                # Avoid obstacles
                usX = 0.0
                usY = 0.0
                for obs in self.obstacles_:
                    
                    psi = self.psi_function(obs, alpha_obs, udX, udY)
                    
                    # Errors
                    e_x = self.x_ - obs.x_
                    e_y = self.y_ - obs.y_
                    
                    # Safety controls
                    if psi < 0:
                        usX += -0.5*(e_x/(e_x**2 + e_y**2))*psi
                        usY += -0.5*(e_y/(e_x**2 + e_y**2))*psi
                    else:
                        usX += 0
                        usY += 0
                
                # Full control
                udX += usX
                udY += usY
            
            return udX, udY
        
        elif method == 5: # Closed form but modify the weight of the controller
            
            # Weights 
            w_safe = 1.0
            w_desi = 0.8
            
            # Calculate safety controls
            
            udX = w_desi * self.ud_x_
            udY = w_desi * self.ud_y_
            
            # Avoid agents
            usX = 0.0
            usY = 0.0
            if self.neighbors_:
                
                for n in self.neighbors_:
                    
                    psi = self.psi_function(n, alpha_neigh, udX, udY)
                    
                    # Errors
                    e_x = self.x_ - n.x_
                    e_y = self.y_ - n.y_
                    
                    # print(self.x_, n.x_)
                    # print(self.id_, n.id_)
                    # print(e_y)
                    # input()
                    
                    # Safety controls
                    if psi < 0:
                        usX += -0.5*(e_x/(e_x**2 + e_y**2))*psi
                        usY += -0.5*(e_y/(e_x**2 + e_y**2))*psi
                    else:
                        usX += 0                    
                        usY += 0
                                            
                # Control to avoid just neighbors agents
                udX += w_safe * usX
                udY += w_safe * usY
            
            if self.obstacles_:
                # Avoid obstacles
                usX = 0.0
                usY = 0.0
                for obs in self.obstacles_:
                    
                    psi = self.psi_function(obs, alpha_obs, udX, udY)
                    
                    # Errors
                    e_x = self.x_ - obs.x_
                    e_y = self.y_ - obs.y_
                    
                    # Safety controls
                    if psi < 0:
                        usX += -0.5*(e_x/(e_x**2 + e_y**2))*psi
                        usY += -0.5*(e_y/(e_x**2 + e_y**2))*psi
                    else:
                        usX += 0
                        usY += 0
                
                # Full control
                udX += w_safe * usX
                udY += w_safe * usY
            
            return udX, udY
        
        elif method == 6: # Closed form with function h modified
            
            '''
                For this method considere that the control applied in the agent is:
                    u(x, t) = u_d(x, t) + u_s(x, t)
                where u_d is the desired control or objetive control and u_s is the safety control
                
                            In this method we can modify the h function (barrier function) to see how it changes the behaviour
            '''
            # Calculate safety controls
            
            udX = self.ud_x_
            udY = self.ud_y_
            
            # Avoid agents
            usX = 0.0
            usY = 0.0
            if self.neighbors_:
                
                for n in self.neighbors_:
                    
                    psi = self.psi_function(n, alpha_neigh, udX, udY, h_function=4)
                    
                    # Errors
                    e_x = self.x_ - n.x_
                    e_y = self.y_ - n.y_
                    
                    # Safety controls
                    if psi < 0:
                        usX += -0.5*(e_x/(e_x**2 + e_y**2))*psi
                        usY += -0.5*(e_y/(e_x**2 + e_y**2))*psi
                    else:
                        usX += 0                    
                        usY += 0
                                            
                # Control to avoid just neighbors agents
                udX += usX
                udY += usY
            
            if self.obstacles_:
                # Avoid obstacles
                usX = 0.0
                usY = 0.0
                for obs in self.obstacles_:
                    
                    psi = self.psi_function(obs, alpha_obs, udX, udY)
                    
                    # Errors
                    e_x = self.x_ - obs.x_
                    e_y = self.y_ - obs.y_
                    
                    # Safety controls
                    if psi < 0:
                        usX += -0.5*(e_x/(e_x**2 + e_y**2))*psi
                        usY += -0.5*(e_y/(e_x**2 + e_y**2))*psi
                    else:
                        usX += 0
                        usY += 0
                
                # Full control
                udX += usX
                udY += usY
            
            return udX, udY
        
        elif method == 7: # Complete form CLF-CBF-QP
            
            '''
                Here we add the control lyapunov function in the QP constrint to find a solution 
            '''
            
            udX = self.ud_x_
            udY = self.ud_y_
            
            agents_and_obs = []
            
            if self.obstacles_:
                agents_and_obs.extend(self.obstacles_)
            
            if self.neighbors_:
                agents_and_obs.extend(self.neighbors_)
                
            # Calculate barrier values and coeffs in h_dot to avoid agents
            self.calculate_h_and_v_coeffs(agents_and_obs)
                
            # Define the problem data
            self.define_QP_problem_for_CLF_CBF(udX, udY, alpha_obs)
                
            if self.agents_QP_solver_ is None or self.new_QP_:
                    
                # Creat
                self.agents_QP_solver_ = osqp.OSQP()
                self.agents_QP_solver_.setup(self.P_, self.q_, self.A_, self.l_, self.u_, verbose=False, time_limit=0)
                
            else:
                    
                # Update
                self.agents_QP_solver_.update(q=self.q_, l=self.l_, u=self.u_, Ax=self.A_.data)
                    
            # Solve QP problem
            res = self.agents_QP_solver_.solve()
            udX, udY, _ = res.x 
            
            # For infeasible solution
            udX = self.ud_x_ if udX is None else udX
            udY = self.ud_y_ if udY is None else udY
            
            return udX, udY
        
        elif method == 8: # Gradient recentered barrier function scheme
                        
            # Control gains
            c1 = 0.1
            
            Bx = 0.0         
            By = 0.0   
                
            # Agents
            if self.neighbors_:
                
                # Constraints parameters
                delta_inf = self.radius_*1.2
                
                for n in self.neighbors_:
                    
                    xe = self.x_ - n.x_ #- self.disired_distance_
                    ye = self.y_ - n.y_ #- self.disired_distance_
                    d = np.sqrt((xe)**2 + (ye)**2)
                    
                    Bx += -xe/(d*(d - delta_inf)**2) - 1/delta_inf                   
                    By += -ye/(d*(d - delta_inf)**2) - 1/delta_inf
            
            # Obstacles
            if self.obstacles_:

                for obs in self.obstacles_:
                    
                    xe = self.x_ - obs.x_ #- obs.radius_*1.2
                    ye = self.y_ - obs.y_ #- obs.radius_*1.2 
                    d = np.sqrt((xe)**2 + (ye)**2)
                    
                    Bx += -xe/(d*(d - obs.radius_*1.2)**2) - 1/obs.radius_*1.2                    
                    By += -ye/(d*(d - obs.radius_*1.2)**2) - 1/obs.radius_*1.2  
                    
                    # Bx += -(xe)/(d**2 - d*obs.radius_*1.2) #- 1/obs.radius_*1.2 
                    # By += -(ye)/(d**2 - d*obs.radius_*1.2) #- 1/obs.radius_*1.2                   
            
            wGx = (1)*(Bx)
            wGy = (1)*(By)
            
            ux = -c1*wGx + self.ud_x_
            uy = -c1*wGy + self.ud_y_
            
            return ux, uy
        
        elif method == 9: # Weight recentered barrier function scheme
                        
            # Control gains
            c1 = 0.1
            
            Bx = 0.0         
            By = 0.0      
                
            # Agents
            if self.neighbors_:
                
                # Constraints parameters
                delta_inf = self.radius_*1.2
                
                for n in self.neighbors_:
                    
                    xe = self.x_ - n.x_ #- self.disired_distance_
                    ye = self.y_ - n.y_ #- self.disired_distance_
                    d = np.sqrt((xe)**2 + (ye)**2)
                    w = 1 + 1/d
                    
                    Bx += w*(-xe/(d*(d - delta_inf)**2))                  
                    By += w*(-ye/(d*(d - delta_inf)**2))
            
            # Obstacles
            if self.obstacles_:

                for obs in self.obstacles_:
                    
                    xe = self.x_ - obs.x_ #- obs.radius_*1.2
                    ye = self.y_ - obs.y_ #- obs.radius_*1.2 
                    d = np.sqrt((xe)**2 + (ye)**2)
                    w = 1 + 1/d
                    
                    Bx += w*(-xe/(d*(d - obs.radius_*1.2)**2))                   
                    By += w*(-ye/(d*(d - obs.radius_*1.2)**2)) 
                    
                    # Bx += -(xe)/(d**2 - d*obs.radius_*1.2) #- 1/obs.radius_*1.2 
                    # By += -(ye)/(d**2 - d*obs.radius_*1.2) #- 1/obs.radius_*1.2                   
            
            wGx = (1)*(Bx)
            wGy = (1)*(By)
            
            ux = -c1*wGx + self.ud_x_
            uy = -c1*wGy + self.ud_y_
            
            return ux, uy
    
    def HOCBF_action(self, method=3):
        
        '''
            High Order Control Barrier Function (HOCBF) of relative degree "m"
            
            For second order...
        '''
        
        if method == 1:
            
            udX = self.ud_x_
            udY = self.ud_y_
            
            # Obstacles and agents
            obstacles = []
            if self.obstacles_:
                obstacles.extend(self.obstacles_)
            # if self.neighbors_:
            #     obstacles.extend(self.neighbors_)
                
            self.calculate_HO_constraints(obstacles, change=len(self.obstacles_))
            
            self.define_QP_problem_data(udX, udY, 1.0)
            
            if self.agents_QP_solver_ is None or self.new_QP_:
                
                self.agents_QP_solver_ = osqp.OSQP()
                self.agents_QP_solver_.setup(self.P_, self.q_, self.A_, self.l_, self.u_, verbose=False, time_limit=0)
                
            else:
                self.agents_QP_solver_.update(q=self.q_, l=self.l_, u=self.u_, Ax=self.A_.data)
                
            res = self.agents_QP_solver_.solve()
            udX, udY, _ = res.x
            
            # For infeasible solution
            udX = self.ud_x_ if udX is None else udX
            udY = self.ud_y_ if udY is None else udY
            
            return udX, udY
        
        elif method == 2:
            
            # Calculate safety controls
            
            udX = self.ud_x_
            udY = self.ud_y_
            
            # Avoid agents
            usX = 0.0
            usY = 0.0
            if self.neighbors_:
                
                for n in self.neighbors_:
                    
                    psi = self.HO_psi_function(n, udX, udY, b=3, obsType='robot')
                    self.psi_ag.append(psi)
                    
                    # Consider the no reciprocal function
                    ex = self.x_ - n.x_
                    ey = self.y_ - n.y_
                    
                    # Safety control
                    if psi <= 0:
                        usX += -0.5*(ex/(ex**2 + ey**2))*psi
                        usY += -0.5*(ey/(ex**2 + ey**2))*psi
                        # self.break_ = True
                
                udX += usX
                udY += usY
                
            else: 
                try:
                    self.psi_ag.append(self.psi_ag[-1])
                except:
                    self.psi_ag.append(0)
                
            if self.obstacles_:
                
                usX = 0.0
                usY = 0.0
                
                for obs in self.obstacles_:
                    
                    psi = self.HO_psi_function(obs, udX, udY, b=1, obsType='circle')
                    self.psi_obs.append(psi)
                    
                    ex = self.x_ - obs.x_
                    ey = self.y_ - obs.y_
                    
                    if psi <= 0:
                        usX += -0.5*(ex/(ex**2 + ey**2))*psi
                        usY += -0.5*(ey/(ex**2 + ey**2))*psi
                        # self.break_ = True
                        
                udX += usX
                udY += usY
                
            else: 
                try:
                    self.psi_obs.append(self.psi_obs[-1])
                except:
                    self.psi_obs.append(0)
                
            return udX, udY
    
        elif method == 3: # Barrier function (h) with velocities 
            
            # Calculate safety controls
            
            udX = self.ud_x_
            udY = self.ud_y_
            
            # Avoid agents
            usX = 0.0
            usY = 0.0
            if self.neighbors_:
                
                acc_contsraint = True
                
                for n in self.neighbors_:
                    
                    psi = self.HO_psi_function(n, udX, udY, acc_contsraint, b=4, obsType='robot')
                    
                    # Consider the no reciprocal function
                    ex = self.x_ - n.x_
                    ey = self.y_ - n.y_
                    z = np.sqrt(ex**2 + ey**2)
                    
                    # Safety control
                    if psi <= 0:
                        usX += -(z/ex)*psi
                        usY += -(z/ey)*psi
                
                udX += usX
                udY += usY
                
            if self.obstacles_:
                
                acc_contsraint = False
                
                usX = 0.0
                usY = 0.0
                
                for obs in self.obstacles_:
                    
                    psi = self.HO_psi_function(obs, udX, udY, acc_contsraint, b=4, obsType='circle')
                    self.psi_obs.append(psi)
                    
                    ex = self.x_ - obs.x_
                    ey = self.y_ - obs.y_
                    z = np.sqrt(ex**2 + ey**2)
                    
                    if psi <= 0:
                        usX += -(z/ex)*psi
                        usY += -(z/ey)*psi
                        
                udX += usX
                udY += usY
                
            return udX, udY
    
    def calculate_HO_constraints(self, obstacles, change):
        
        '''
            Here get the derivates of the barrier function
        '''
        
        alpha_1_obs = 5.0
        alpha_1_neig = 5.0
        alpha_2_obs = 5.0
        alpha_2_neig = 5.0
                
        x_1 = self.x_
        y_1 = self.y_

        x_2 = self.vel_x_
        y_2 = self.vel_y_
        
        h = []
        coeffs_dhdx = []
                
        for i, obs in enumerate(obstacles):
            
            if i < change:
                alpha_1 = alpha_1_obs
                alpha_2 = alpha_2_obs
            else:
                alpha_1 = alpha_1_neig
                alpha_2 = alpha_2_neig
            
            x_o = obs.x_
            y_o = obs.y_
            
            psi_0 = (x_1 - x_o)**2 + (y_1 - y_o)**2 - obs.radius_**2
            psi_1 = 2*(x_1 - x_o)*x_2 + 2*(y_1 - y_o)*y_2 + alpha_1*psi_0
            
            try:
                self.psi_0[i].append(psi_0)
                self.psi_1[i].append(psi_1)
            except:
                self.psi_0[i] = [psi_0]
                self.psi_1[i] = [psi_1]
            
            b_dot = 2*(x_1 - x_o)*x_2 + 2*(y_1 - y_o)*y_2
            h_aux = 2*x_2*2 + 2*y_2**2 + alpha_1*(b_dot) + alpha_2*( b_dot + alpha_1*psi_0 )
            # h_aux = 2*x_2*2 + 2*y_2**2 + (b_dot)**2 + ( b_dot + psi_0**2 )**2
            
            coeffs_aux = [2*(x_1 - x_o), 2*(y_1 - y_o)] + [1]
            
            h.append(h_aux)
            coeffs_dhdx.append(coeffs_aux)
            
        self.h_ = h
        self.coeffs_dhdx_ = coeffs_dhdx
    
    def calculate_h_and_coeffs(self, obstacles):
        
        '''
            Here we are gonna calculate the "h" function and differential coeffitients
                h -> norm(x - xobs - desired)^2
        '''

        h = [] # Barrier values (here, remaining distance to each obstacle)
        coeffs_dhdx = [] # dhdt = dhdx * dxdt = dhdx * u
        
        for obs in obstacles:
                
            haux = (self.x_ - obs.x_)**2 + (self.y_ - obs.y_)**2 - (obs.radius_*1.2)**2
            h.append(haux)
                
            coeffs_dhdx_aux = [2*(self.x_ - obs.x_), 2*(self.y_ - obs.y_)] + [1]
            coeffs_dhdx.append(coeffs_dhdx_aux)
            
        self.h_ = h
        self.coeffs_dhdx_ = coeffs_dhdx
        
    def calculate_h_and_v_coeffs(self, obstacles):
        
        '''
            Calculate the barrier function h (or B) differential coeffitients and 
            lyapunov function V coeffitiens.
        '''
        
        h_ = []
        h_coeffs = []
        v_ = []
        v_coeffs = []
        
        desiredX = self.leader_.x_ + self.desired_position_[0]
        desiredY = self.leader_.y_ + self.desired_position_[1]
        
        errorX = self.x_ - desiredX
        errorY = self.y_ - desiredY
        
        desiredX = -0.0005*errorX 
        desiredY = -0.0005*errorY 
        
        for obs in obstacles:
            
            haux = (self.x_ - obs.x_)**2 + (self.y_ - obs.y_)**2 - (obs.radius_*1.2)**2
            h_.append(haux)
            
            coef = [2*(self.x_ - obs.x_), 2*(self.y_ - obs.y_)] + [1]
            h_coeffs.append(coef)
            
            vaux = (self.x_ - desiredX)**2 + (self.y_ - desiredY)**2
            v_.append(vaux)
            
            coef = [2*(self.x_ - desiredX), 2*(self.y_ - desiredY)] + [1]
            v_coeffs.append(coef)
            
        self.h_ = h_
        self.v_ = v_
        self.coeffs_dhdx_ = h_coeffs
        self.coeffs_dvdx_ = v_coeffs
        
    def define_QP_problem_data(self, udX, udY, alpha, penaltySlack=2.5):
        
        '''
            P: shape (nx, nx)
            q: shape (nx,)
            A: shape (nh+nx, nx)
            l: shape (nh+nx,)
            u: shape (nh+nx,)
            
            nx -> number of states
            nh -> number of control barrier functions
        '''
        
        P = sparse.csc_matrix([[1, 0, 0], [0, 1, 0], [0, 0, penaltySlack]])
        q = np.array([-udX, -udY, 0])
        A = sparse.csc_matrix([c for c in self.coeffs_dhdx_] + [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        l = np.array([-alpha*h for h in self.h_] + [np.minimum(udX, 0), np.minimum(udY, 0), 0])
        u = np.array([np.inf for _ in self.h_] + [np.maximum(udX, 0), np.maximum(udY, 0), np.inf])
        # u = np.array([10 for _ in self.h_] + [np.maximum(udX, 0), np.maximum(udY, 0), 10])
        
        self.P_ = P 
        self.q_ = q 
        self.A_ = A 
        self.l_ = l 
        self.u_ = u
        
    def define_QP_problem_for_CLF_CBF(self, udX, udY, alpha, penaltySlack=1.0):
        
        '''
            P: shape (nx, nx)
            q: shape (nx,)
            A: shape (nh+nx, nx)
            l: shape (nh+nx,)
            u: shape (nh+nx,)
            
            nx -> number of states
            nh -> number of control barrier functions
        '''
        
        epsilon = 0.0001
        delta = 0.0
        
        P = sparse.csc_matrix([[1, 0, 0], [0, 1, 0], [0, 0, penaltySlack]])
        # q = np.array([delta**2, delta**2, 0])
        q = np.array([-udX, -udY, 0])
        
        Ap = []
        Ap.extend(self.coeffs_dhdx_)
        Ap.extend(self.coeffs_dvdx_)
        A = sparse.csc_matrix(Ap + [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        lp = []
        for h in self.h_:
            lp.append(-alpha*h)
        for _ in self.v_:
            lp.append(-np.inf)
        l = np.array(lp + [np.minimum(udX, 0), np.minimum(udY, 0), 0])
        
        up = []
        for _ in self.h_:
            up.append(np.inf)
        for v in self.v_:
            up.append(-epsilon*v + delta)
        u = np.array(up + [np.maximum(udX, 0), np.maximum(udY, 0), np.inf])
        
        self.P_ = P 
        self.q_ = q 
        self.A_ = A 
        self.l_ = l 
        self.u_ = u
    
    def psi_function(self, obs, alpha, udX, udY, h_function = 1):
        
        '''
            This function is used to calculate de avoidance velocity in closed form
                
        '''
        
        # Data
        u_x = udX
        u_y = udY
        x = self.x_
        y = self.y_
        x_o = obs.x_
        y_o = obs.y_
        d = obs.radius_*1.2
        
        if h_function == 1:
            
            h_dot = 2*(x - x_o)*u_x + 2*(y - y_o)*u_y
            h = (x - x_o)**2 + (y - y_o)**2 - d**2
            
        elif h_function == 2:
            
            arg = np.sqrt((x - x_o)**2 + (y - y_o)**2) - d
            arg_dot = ((x - x_o)*u_x + (y - y_o)*u_y)/np.sqrt((x - x_o)**2 + (y - y_o)**2)
            h = -np.log(arg)
            h_dot = arg_dot/arg
            
            # h = 1/h
            
        elif h_function == 3:
            
            arg = np.sqrt((x - x_o)**2 + (y - y_o)**2) - d
            arg_dot = -(2*(x - x_o)*u_x + 2*(y - y_o)*u_y)/np.sqrt((x - x_o)**2 + (y - y_o)**2)
            h = 1/arg
            h_dot = arg_dot/arg**2
            
            # h = 1/h
            
        elif h_function == 4:
            
            arg = (x - x_o)**2 + (y - y_o)**2 - d**2
            arg_dot = 2*(x - x_o)*u_x + 2*(y - y_o)*u_y
            h = 1/arg
            h_dot = arg_dot/arg**2
            
            h = 1/h
            
        elif h_function == 5:
            
            arg = (x - x_o)**2 + (y - y_o)**2 - d**2
            # arg_dot = 2*(x - x_o)*(2*d**2 + 1)*u_x + 2*(y - y_o)*(2*d**2 + 1)*u_y
            arg_dot = 2*(x - x_o)*u_x + 2*(y - y_o)*u_y
            h = -np.log( arg/(1+arg) )
            h_dot = -arg_dot/(arg**2 + arg)
            
            h = 1/h
        
        return h_dot + alpha*h
    
    def HO_psi_function(self, n, udX, udY, accConst, b=3, obsType='circle', alphas=[5.0, 5.0]):
        
        '''
            Psi function for high order method
            Select the barrier function
        '''
        
        # # Hurwitz
        # F = np.array([[0, 1], [0, 0]])
        # G = np.array([[0], [1]])
        # K = np.array([[alphas[0], 0], [0, alphas[1]]])
        
        # cl = F - G@K
        
        if obsType == 'circle':
            d = n.radius_ + 2*self.radius_
            
        elif obsType == 'robot':
            d = n.radius_
            
        else:
            raise ValueError('Obstacle type error, It must be "circle" or "robot"')
        
        # Data
        x = self.x_
        y = self.y_
        xo = n.x_
        yo = n.y_
        
        d = n.radius_ + self.radius_
        
        x_vel = self.vel_x_
        y_vel = self.vel_y_
        
        if b == 1: # For the first barrier function (distance)
            
            h = (x - xo)**2 + (y - yo)**2 - d**2
            h_dot = 2*(x - xo)*x_vel + 2*(y - yo)*y_vel
            h_ddot = 2*x_vel**2 + 2*y_vel**2 + 2*(x - xo)*udX + 2*(y - yo)*udY
            
            # return h_ddot + alphas[1]*(h_dot + alphas[0]*h) + alphas[0]*(h_dot)
            return h_ddot + alphas[0]*h_dot + alphas[1]*(h_dot + alphas[0]*h) 
            # return h_ddot + (h_dot + h**3)**3 + (h_dot)**3
            
        elif b == 2: # For the first barrier function (reciprocal)
            
            ex = x - xo
            ey = y - yo
            
            arg = np.sqrt(ex**2 + ey**2) - d
            arg_dot = -d * np.sqrt(ex**2 + ey**2) + ex**2 + ey**2
            arg_ddot_x = ex*(xo*x_vel + yo*y_vel - x*x_vel - y*y_vel)*(2 - (d)/(arg+d)) + x_vel*(ex**2 + ey**2 - d*(arg+d))
            arg_ddot_y = ey*(xo*x_vel + yo*y_vel - x*x_vel - y*y_vel)*(2 - (d)/(arg+d)) + y_vel*(ex**2 + ey**2 - d*(arg+d))
            
            h = -np.log(arg)
            h_dot = -(x*x_vel + y*y_vel)/arg_dot
            h_ddot = -(arg_ddot_x*udX + arg_ddot_y*udY)/arg_dot**2
            
            # return h_ddot + alphas[0]*h_dot + alphas[1]*(h_dot + alphas[0]*h) 
            return h_ddot + (h_dot)**3 + (h_dot + (h)**3)**3
        
        elif b == 3: # Robust recirpocal function
            
            ex = x - xo
            ey = y - yo
            
            arg = ex**2 + ey**2 - d**2
            arg_dot = 2*ex*(2*d**2 + 1)*x_vel + 2*ey*(2*d**2 + 1)*y_vel
            arg_ddot_x = 2*(2*d**2)*(-2*ex*(2*arg + 1))*(xo*x_vel + yo*y_vel - x*x_vel - y*y_vel) - x_vel*(arg**2 + arg)
            arg_ddot_y = 2*(2*d**2)*(-2*ey*(2*arg + 1))*(xo*x_vel + yo*y_vel - x*x_vel - y*y_vel) - y_vel*(arg**2 + arg)
            
            h = -np.log( arg/(1+arg) )
            h_dot = -arg_dot/(arg**2 + arg)
            h_ddot = (arg_ddot_x*udX + arg_ddot_y*udY)/(arg**2 + arg)**2
            
            # h = 1/h
            # h_dot = 1/h_dot
            
            # return h_ddot + (h_dot)**3 + (h_dot + (h)**3)**3
            return h_ddot + alphas[0]*(h_dot) + alphas[1]*(h_dot + alphas[0]*(h))
            # return h_ddot + alphas[0]*(1/h_dot) + alphas[1]*(h_dot + alphas[0]*(1/h))
            
        elif b == 4: # Barrier function (h) with velocities
            
            if not accConst:
                accMax_o = 0.0
            
            else:
                accMax_o = n.acc_max_
                
            accReac = self.acc_max_ + accMax_o
            
            xo_vel = n.vel_x_
            yo_vel = n.vel_y_
            
            ex = x - xo 
            ey = y - yo 
            dist = np.sqrt(ex**2 + ey**2)
            
            ex_vel = x_vel - xo_vel
            ey_vel = y_vel - yo_vel
            
            h = np.sqrt(2*accReac*(dist - d)) + (ex*ex_vel + ey*ey_vel)/dist
            
            # Partial derivatives
            dx = (np.sqrt(2)*ex*accReac)/(2*dist*np.sqrt(accReac*(dist - d))) - (ex*(ex*ex_vel + ey*ey_vel))/(dist**3) + (ex_vel/dist)
            dy = (np.sqrt(2)*ey*accReac)/(2*dist*np.sqrt(accReac*(dist - d))) - (ey*(ex*ex_vel + ey*ey_vel))/(dist**3) + (ey_vel/dist)
            dvx = ex/dist
            dvy = ey/dist
            
            h_dot = dx*ex_vel + dy*ey_vel + dvx*udX + dvy*udY
            
            return h_dot + 80.0*h
    
    def detect_collsion(self, d=1.0):
        
        '''
            Here the agent detects the obstacles that are close
                d -> distance detection, before collision
        '''
        
        self.is_collided_ = False
        if self.obstacles_:
            
            for obs in self.obstacles_:
                
                dist = np.linalg.norm(np.array([self.x_, self.y_]) - np.array([obs.x_, obs.y_]))
                if dist <= obs.radius_*2:
                    self.is_collided_ = True
                
    def detect_neighbors(self, agents, range=4):
        
        '''
            This function detec and add to consider agents in avoidance
                agents -> list of all agents
        '''
        
        if len(agents) >= 2:
            
            close = []
            for agent in agents:
                
                dist = np.linalg.norm(np.array([self.x_, self.y_]) - np.array([agent.x_, agent.y_]))
                if dist <= range and self.id_ != agent.id_:
                    close.append(agent)
                    
            self.neighbors_ = close
            self.new_QP_ = True
            
        else:
            self.new_QP_ = False
    
    def save(self, agents=None):
        
        '''
            Save all de state of the agent and the observer
        '''
        
        if self.leader_role_:
            
            self.data_pos_x_.append(self.x_)
            self.data_pos_y_.append(self.y_)
            self.data_vel_x_.append(self.vel_x_)
            self.data_vel_y_.append(self.vel_y_)
            self.data_acc_x_.append(self.acc_x_)
            self.data_acc_y_.append(self.acc_y_)
            
            return
        
        elif self.observer_order_ == 1:
            
            self.data_pos_x_.append(self.x_)
            self.data_pos_y_.append(self.y_)
            self.data_vel_x_.append(self.vel_x_)
            self.data_vel_y_.append(self.vel_y_)
            self.data_acc_x_.append(self.acc_x_)
            self.data_acc_y_.append(self.acc_y_)
            
            self.data_desired_vel_x_.append(self.ud_x_)
            self.data_desired_vel_y_.append(self.ud_y_)
            self.data_safety_vel_x_.append(self.us_x_)
            self.data_safety_vel_y_.append(self.us_y_)
                
            self.data_l_pos_x_.append(self.l_pos_x_)
            self.data_l_pos_y_.append(self.l_pos_y_)
            self.data_l_vel_x_.append(self.l_vel_x_)
            self.data_l_vel_y_.append(self.l_vel_y_)
            
            for agent in agents:
                if self.id_ != agent.id_:
                    
                    dist = np.sqrt( (self.x_ - agent.x_)**2 + (self.y_ - agent.y_)**2 )
                    self.data_neighbors_distances_[agent.id_].append(dist)
            
            if self.obstacles_:
                for obs in self.obstacles_:
                    
                    dist = np.sqrt( (self.x_ - obs.x_)**2 + (self.y_ - obs.y_)**2 )
                    self.data_obstacles_distances_[obs.tag_].append(dist)
            
        elif self.observer_order_ == 2:
            
            self.data_pos_x_.append(self.x_)
            self.data_pos_y_.append(self.y_)
            self.data_vel_x_.append(self.vel_x_)
            self.data_vel_y_.append(self.vel_y_)
            self.data_acc_x_.append(self.acc_x_)
            self.data_acc_y_.append(self.acc_y_)
            
            self.data_desired_vel_x_.append(self.ud_x_)
            self.data_desired_vel_y_.append(self.ud_y_)
            self.data_safety_vel_x_.append(self.us_x_)
            self.data_safety_vel_y_.append(self.us_y_)
                
            self.data_l_pos_x_.append(self.l_pos_x_)
            self.data_l_pos_y_.append(self.l_pos_y_)
            self.data_l_vel_x_.append(self.l_vel_x_)
            self.data_l_vel_y_.append(self.l_vel_y_)
            self.data_l_acc_x_.append(self.l_acc_x_)
            self.data_l_acc_y_.append(self.l_acc_y_)
            
# Barrier functions
def BFC_data(zx, zy, case):
    
    d = np.linalg.norm(np.array([1.5, 1.5]))
    deltaInf = 0.5
    deltaSup = 4.0
    
    k1 = (deltaInf**2)/((d**2)*((d**2) - deltaInf**2))
    k2 = 1/((deltaSup**2) - d**2)
    
    B = []
    for i in range(len(zx)):
        
        z = np.linalg.norm(np.array([zx[i], zy[i]]))

        if case == 1:
            B.append(k1*( np.log( (deltaSup**2)/((deltaSup**2) - z**2) ) - np.log( (deltaSup**2)/((deltaSup**2) -
             d**2) ) ) + k2*( np.log( (z**2)/((z**2) - deltaInf**2) ) - np.log( (d**2)/((d**2) - deltaInf**2) ) ))
            
        elif case == 2: # Not lost the connection
            B.append(k1*( np.log( (deltaSup**2)/((deltaSup**2) - z**2) ) - np.log( 
                                        (deltaSup**2)/((deltaSup**2) - d**2) ) ))
            
        elif case == 3: # Collision avoidance
            B.append(k2*( np.log( (z**2)/((z**2) - deltaInf**2) ) - np.log( (d**2)/((d**2) - deltaInf**2) ) ))
        
    return B

def BFC_data_mod(zx, zy, case):
    
    d = np.linalg.norm(np.array([1.5, 1.5]))
    deltaInf = 0.5
    deltaSup = 4.0
    
    k1 = (deltaInf**2)/((d**2)*((d**2) - deltaInf**2))
    k2 = 1/((deltaSup**2) - d**2)
    
    B = []
    for i in range(len(zx)):
        
        z = np.linalg.norm(np.array([zx[i], zy[i]]))

        if case == 1:
            B.append(k1*( np.log( deltaSup/((deltaSup**2) - z**2) ) - np.log( deltaSup/((deltaSup**2) -
             d**2) ) ) + k2*( np.log( z/((z**2) - deltaInf**2) ) - np.log( z/((d**2) - deltaInf**2) ) ))
            
        elif case == 2: # Not lost the connection
            B.append(k1*( np.log( 1/((deltaSup**2) - z**2) ) - np.log( 
                                        1/((deltaSup**2) - d**2) ) ))
            
        elif case == 3: # Collision avoidance
            B.append(k2*( np.log( z/((z**2) - deltaInf**2) ) - np.log( z/((d**2) - deltaInf**2) ) ))
        
    return B