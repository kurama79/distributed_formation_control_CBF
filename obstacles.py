'''
    This scripts contains the obstacle class
'''

import numpy as np
from shapely.geometry import Polygon, Point

class Obstacle:
    
    def __init__(self, i, currentObstacles, minR, maxR, randomly, upperBound, lowerBound, dynamic=False):
        
        self.tag_ = i
        self.dynamic_ = dynamic 
        self.radius_ = np.random.uniform(minR, maxR)
        
        self.define_position(currentObstacles, randomly, upperBound, lowerBound)
        self.define_velocity()
        
        # self.define_acceleration()
        
    def define_position(self, obstacles, rand, upB, lowB):
        
        '''
            In this function we initialize the obstacle position considering if exist more obstacles
                obstacles -> list of obstacles generated
                rand      -> bool value if the obstacle configuration will be generated randomly
        
        '''
        
        if rand:
            
            if not obstacles:
                
                x = np.random.uniform(lowB, upB)
                y = np.random.uniform(lowB, upB)
                
            else:
                
                free = False # Consider if the obstacle intersect others
                while not free:
                    
                    x = np.random.uniform(lowB, upB)
                    y = np.random.uniform(lowB, upB)
                    newObs = Point(x, y).buffer(self.radius_*2.0)
                    
                    for obs in obstacles:
                        
                        currentObs = Point(obs.x_, obs.y_).buffer(obs.radius_)
                        
                        if newObs.intersects(currentObs):
                            
                            free = False
                            
                            del currentObs
                            break 
                        
                        else:
                            
                            free = True
                            del currentObs
                            
                    del newObs
                    
        else:
            
            x = 0            
            y = 5            
        
        self.x_ = x
        self.y_ = y
        
    def define_velocity(self):
        
        if not self.dynamic_:
            
            self.vel_x_ = 0.0
            self.vel_y_ = 0.0