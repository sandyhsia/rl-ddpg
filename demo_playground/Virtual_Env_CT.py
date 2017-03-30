import time
import pygame
import sys 
import numpy as np
import math
import random as random
from pygame.locals import *
import cv2
import time

# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'DEMO'
EPISODE = 10000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 1 # The number of experiment test every 100 episode
color = 0, 102, 102
action_space = ['w', 'a', 'd']
default_bounding_lines = [[1., 0., -640.],
                                [1., 0., 0.],
                                [0., 1., -480.],
                                [0., 1., 0.,]]

simple_border = [[30, 30], [30, 70],
                            [30, 70], [70, 70],
                            [70, 70], [70, 30],
                            [70, 30], [30, 30]]

#border = [[20, 30], [15, 30],
#               [15, 30], [50, 40],
#              [50, 40], [20, 30]]
border= [[]]

limited_eyesight = 1
eyesight = 25

with_cam = 0
speed_level0 = 5
speed_level1 = 10
turn_level0 = math.pi/8
turn_level1 = math.pi/4
car_body_lenth = 8
car_body_width = 4

default_car_center = [30., 20.]
default_angle = 0.
DEBUG_MODE = 0
vary_border_cnt = 1     # 0: no border
                                      # 1: simple border
                                      # 2: various border
driving_route = [[15, 80], [80, 80], [80, 15], [15, 15], [15, 80], [185, 80], [185, 185], [15, 185]]
car_agent_num = 1


class Virtual_Env():
        # Virtural Env + Actual Env (captured from camera)
        def __init__(self, ENV_NAME, w, h): 
                # maybe tell me how many cars in gray_dst some time later
                # might related to observation space

                pygame.init()
                # print "w, h:", w, h
                self.screen = pygame.display.set_mode((w, h))
                pygame.display.set_caption(ENV_NAME)
                pygame.key.set_repeat(50)
                self.screen.fill(color)

                self.action_space = action_space
                self.observation_space = np.zeros(360)
                self.bounding_lines = default_bounding_lines # default
                self.bounding_lines[0][2] = -w
                self.bounding_lines[2][2] = -h
                # self.bounding_cnt = np.array([border[0], border[2], border[4]], dtype=np.int)
                self._with_cam = with_cam
                self.w = w
                self.h = h
                self.default_car_center = [15., 15.]
                self.default_angle = 0.
                self.car_target_pt = [0., 0.]
                self.eat_up_time = 0
                self.driving_route = driving_route
                self.route_pt_counter = 0


        def reset(self, mode):
                if mode == 0: # all init
                        
                        if vary_border_cnt > 0:
                                self.set_up_border(4)
                                while vary_border_cnt == 2 and (cv2.contourArea(self.bounding_cnt) > (0.05* self.w * self.h) or cv2.contourArea(self.bounding_cnt) < (0.005* self.w * self.h)):
                                        self.set_up_border(4)

                        self.car_center = [15., 15.]
                        self.car_target_pt = self.driving_route[0]
                        self.route_pt_counter = 1
                        self.angle = 270.
                        self.eat_up_time = 0
                         
                elif mode == 1: # init a new target pt

                        self.car_target_pt = self.driving_route[self.route_pt_counter]
                        self.route_pt_counter = (self.route_pt_counter+2)%4

                print "reset car location:", self.car_center, self.angle
                print "reset car target location:", self.car_target_pt
                state, reward, done = self.step(self.car_center, self.angle, (0, 0))
                return state

        def step(self, car_center, angle, action):

                self.screen.fill(color)
                center = [0., 0.]
                center = car_center
                if self._with_cam == 0:
                        car_center, angle, done = self.agent_action(car_center, angle, action)
                        # print car_center, angle, done
                        self.car_center = car_center
                        self.angle = angle
                        # print car_center, angle, done
                else:
                        pass

                if self._with_cam == 0 and done == True:
                        self.display_border()
                        pygame.display.update()
                        return np.zeros(362), -20, done

                else:
                        self.display_border()
                        pygame.display.update()

                        solution = np.zeros((360, 2))
                        intersect_points_vec = np.zeros((360,2))
                        distance_vec = np.zeros(360)
                        bounding_lines = self.bounding_lines
                        center = self.car_center

                        if vary_border_cnt > 0:
                                border = self.border

                        
                        if limited_eyesight == 1:
                                for i in range(360):
                                        intersect_points_vec[i] = (center[0] + eyesight*np.cos((-i+180)*math.pi/180), center[1] - eyesight*np.sin((-i+180)*math.pi/180))
                                        distance_vec[i] = eyesight

                                '''for i in range(360):
                                        if(i%15== 0):
                                                pygame.draw.line(self.screen, [164,125,255], center, intersect_points_vec[(180-int(angle)+i)%360], 1)
                                pygame.display.update()
                                time.sleep(0.5)'''


                        '''k_arr = np.zeros(360)
                        line_param_arr = np.ones((360, 2))
                        c = np.zeros(360)
                        for i in range(360):
                                if i != 180 and i != 0:
                                        k_arr[i] = math.tan((i-180)*(math.pi)/180)
                                        line_param_arr[i] = (1.00, -1/k_arr[i])
                                        c[i] = np.dot(center, line_param_arr[i].T)
                                else:
                                        k_arr[i] = 0
                                        line_param_arr[i] = (0.00, 1.00)
                                        c[i] = np.dot(center, line_param_arr[i].T)

                        # print "bounding_lines len:", len(bounding_lines)'''

                        for j in range(len(bounding_lines)):

                                
                                if j == 0:
                                        line_segament = [[self.w, 0], [self.w, self.h]]
                                elif j == 1:
                                        line_segament = [[0, 0], [0, self.h]]
                                elif j == 2:
                                        line_segament = [[0, self.h], [self.w, self.h]]
                                elif j == 3:
                                        line_segament = [[0, 0], [self.w, 0]]
                                elif j > 3:
                                        line_segament = [self.border[(j-4)*2+0], self.border[(j-4)*2+1]]
                                # print "line_seg", line_segament

                                distance_vec, intersect_points_vec = self.solve_distance_vec(distance_vec, intersect_points_vec, center, line_segament, j)

                                '''for i in range(360):
                                        if(i%15== 0):
                                                pygame.draw.line(self.screen, [164,125,255], center, intersect_points_vec[(180-int(angle)+i)%360], 1)
                                pygame.display.update()
                                time.sleep(0.5)'''

                                '''for i in range(360):
                                        param = np.zeros((2, 2))
                                        param[0] = line_param_arr[i]
                                        param[1] = (bounding_lines[j][0], bounding_lines[j][1])
                                        bias = (c[i], -bounding_lines[j][2])
                                        if param[0][0]*param[1][1] != param[0][1] * param[1][0]:
                                                solution[i] = np.linalg.solve(param, bias)
                                        if(solution[i][0] >=self.w):
                                                solution[i][0] = self.w
                                        if(solution[i][0] <=0):
                                                solution[i][0] =0
                                        if(solution[i][1] >= self.h):
                                                solution[i][1] = self.h
                                        if(solution[i][1] <= 0):
                                                solution[i][1] = 0
                                
                                if j <= 3:
                                        for i in range(360):
                                                if int(vector_direction(center, solution[i])) == (i-180)*(-1):
                                                        if intersect_points_vec[i][0] == 0 and intersect_points_vec[i][1] == 0:
                                                                intersect_points_vec [i] = solution[i]
                                                                distance_vec[i] = two_point_distance(center, intersect_points_vec[i])
                                                        elif two_point_distance(center, solution[i]) < distance_vec[i]:
                                                                intersect_points_vec[i] = solution[i]
                                                                distance_vec[i] = two_point_distance(center, intersect_points_vec[i])

        
                                else:
                                        for i in range(360):
                                                if int(vector_direction(center, solution[i])) == (i-180)*(-1) and abs(vector_direction(border[(j-4)*2+0], solution[i]) - vector_direction(border[(j-4)*2+1], solution[i])) == 180:
                                                        if two_point_distance(center, solution[i]) < distance_vec[i]:
                                                                intersect_points_vec[i] = solution[i]
                                                                distance_vec[i] = two_point_distance(center, intersect_points_vec[i])'''



                        for i in range(0, 30):
                              if(i%5== 0):
                                        pygame.draw.line(self.screen, [164,125,255], center, intersect_points_vec[(180-int(angle)+i)%360], 1)
                        pygame.display.update()

                        for i in range(330, 360):
                              if(i%5== 0):
                                        pygame.draw.line(self.screen, [164,125,255], center, intersect_points_vec[(180-int(angle)+i)%360], 1)
                        pygame.display.update()



                        ''' Give next_state, reward, done 
                            to agent '''
                        next_distance_vec = distance_vec
                        for i in range(360):
                                next_distance_vec[i] = distance_vec[(180-int(angle)+i)%360]

                        next_distance_vec = np.append(next_distance_vec, two_point_distance(center, self.car_target_pt))
                        
                        target_direction = vector_direction(center, self.car_target_pt)
                        if angle >= 0 and angle < 180:
                                angle_to_append = angle
                        elif angle >= 180 and angle <=360:
                                angle_to_append = angle - 360

                        if target_direction - angle_to_append >180:
                                angle_diff = 360 - (target_direction - angle_to_append)
                        elif target_direction - angle_to_append < -180:
                                angle_diff = 360 +(target_direction - angle_to_append)
                        else:
                                angle_diff = target_direction-angle_to_append
                        next_distance_vec = np.append(next_distance_vec, (angle_diff))

                        head = [0., 0.]
                        angle_in_rad = angle*math.pi/180
                        head[0] = center[0] + car_body_lenth*np.cos(angle_in_rad)
                        head[1] = center[1] - car_body_lenth*np.sin(angle_in_rad)
                        if two_point_distance(head, self.car_target_pt) <= 10 or two_point_distance(center, self.car_target_pt) <= 10:
                                reward = 20
                                print "Agent: Yoho!!"
                                self.eat_up_time += 1
                                next_distance_vec = self.reset(1)
                                done = False

                                if DEBUG_MODE == 1:
                                        print "next_distance_vec[0:10]",  next_distance_vec[0:10]
                                        print "target direction", vector_direction(center, self.car_target_pt), "angle", angle
                                        print "dis:", next_distance_vec[360], "angle diff:", next_distance_vec[361] 
                                        print "reward", reward
                                        time.sleep(1)

                                return next_distance_vec, reward, done

                        if done == False:
                                reward = self.reward_method(next_distance_vec, action)
                                if DEBUG_MODE == 1:
                                        print "next_distance_vec[0:10]",  next_distance_vec[0:10]
                                        print "target direction", vector_direction(center, self.car_target_pt), "angle", angle
                                        print "dis:", next_distance_vec[360], "angle diff:", next_distance_vec[361] 
                                        print "reward", reward
                                        time.sleep(1)

                                return next_distance_vec, reward, done

        def agent_action(self, car_center, angle, action):

                head = [0., 0.]
                tail = [0., 0.]
                speed = action[0]
                steer = action[1]
                angle_in_rad = (angle+steer)*math.pi/180
                center = car_center
                
                center[0] += speed *np.cos(angle_in_rad)
                center[1] -= speed*np.sin(angle_in_rad)
                

                head[0] = center[0] + car_body_lenth*np.cos(angle_in_rad)
                head[1] = center[1] - car_body_lenth*np.sin(angle_in_rad)
                tail[0] = center[0] + car_body_lenth*np.cos(angle_in_rad + math.pi)
                tail[1] = center[1] - car_body_lenth*np.sin(angle_in_rad + math.pi)
                done = self.display_check_agent(center, angle_in_rad, head, tail, 1, self.car_target_pt, 1)


                return_angle = int(180*angle_in_rad/math.pi)%360
                return_car_center = car_center
                

                return car_center, return_angle, done

        def reward_method(self, next_state, this_state_action):

                if next_state[360] >0:
                        reward = 5/next_state[360]
                        if reward >5:
                                reward = 5
                else:
                        reward = 5

                if next_state[361] != 0:
                        reward += 5/(abs(next_state[361]))
                        if 5/(abs(next_state[361])) >5:
                                reward = reward - 5/(abs(next_state[361])) + 5
                else:
                        reward += 5

                reward += (abs(this_state_action[0])*math.cos(next_state[361]*math.pi/180) - abs(abs(this_state_action[0])*math.sin(next_state[361]*math.pi/180)))
                # print "extra bonus", (abs(this_state_action[0])*math.cos(next_state[361]*math.pi/180) - abs(abs(this_state_action[0])*math.sin(next_state[361]*math.pi/180)))

                space = 0
                action = this_state_action
                if abs(this_state_action[0]) < 0.5:
                        reward -= 1
                reward += this_state_action[0]/5
                if reward > 10:
                        reward = 10
                return reward

        def set_up_border(self, cnt_pt_num):

                self.reset_border()
                if DEBUG_MODE:
                    print "bounding lines:", self.bounding_lines
                    time.sleep(1)

                if cnt_pt_num == 0 or vary_border_cnt == 0:
                        self.border = [[]]
                        return

                else:
                        if vary_border_cnt == 2:
                                border_lenth = cnt_pt_num*2
                                border = [[0., 0.]]
                                self.border = [[0., 0.]]

                                for i in range(cnt_pt_num):
                                        if i == cnt_pt_num - 1:
                                                border.append([random.random()*self.w, random.random()*self.h])
                                                border[0] = border[len(border) - 1]
                                        else:
                                                border.append([random.random()*self.w, random.random()*self.h])
                                                border.append(border[len(border) - 1])
                        elif vary_border_cnt == 1:
                                border = simple_border

                        for i in range(len(border)/2):
                                if border[i*2+0][1] == border[i*2+1][1]:
                                        self.bounding_lines.append([0., 1., -border[i*2+0][1]])

                                elif border[i*2+0][0] == border[i*2+1][0]:
                                        self.bounding_lines.append([1., 0., -border[i*2+0][0]])

                                else:
                                        arr1 = [[border[i*2+0][1], 1],
                                        [border[i*2+1][1], 1]]
                                        arr2 = [-border[i*2+0][0], -border[i*2+1][0]]
                                        border_line = np.linalg.solve(arr1, arr2)
                                        self.bounding_lines.append([1., border_line[0], border_line[1]])

                        self.border = border
                        self.bounding_cnt = np.array([border[i*2] for i in range(cnt_pt_num)], dtype=np.int)

                        return

        def reset_border(self):

                if len(self.bounding_lines) > 4:
                    init_lenth = len(self.bounding_lines)

                    for i in range(init_lenth - 4):
                            self.bounding_lines.pop()
                            # print i, self.bounding_lines

                    self.bounding_lines = default_bounding_lines # default
                    self.bounding_lines[0][2] = -self.w
                    self.bounding_lines[2][2] = -self.h

                return

        def display_border(self):
                
                if vary_border_cnt > 0:
                        # print "lenth of border:", len(self.border)
                        pygame.draw.polygon(self.screen, [0,0,0], [self.border[i*2+0] for i in range(len(self.border)/2)])

                        #for i in range(len(self.border)/2):
                        #       pygame.draw.line(self.screen, [0,0,0], self.border[i*2+0], self.border[i*2+1], 3)

                return

        def display_check_agent(self, center, angle_in_rad, head, tail, agent_mode, car_target_pt, target_pt_mode):
                if agent_mode == 0: # snake shape
                        pygame.draw.line(self.screen, [255,0,0], head, tail, 3)
                        pygame.draw.circle(self.screen, [0,0,0], (int(head[0]),int(head[1])), 1)
                        
                        if target_pt_mode == 1: # draw
                                pygame.draw.circle(self.screen, [255,0,0], (int(self.car_target_pt[0]),int(self.car_target_pt[1])), 2)
                       
                        pygame.display.update()

                        if head[0] <= self.w and head[0] >= 0 and head[1] <= self.h and head[1] >= 0 and tail[0] <= self.w and tail[0] >= 0 and tail[1] <= self.h and tail[1] >=0:
                                done = False
                        else:
                                done = True
                                print "out of playground."

                        if vary_border_cnt>0 and  cv2.pointPolygonTest(self.bounding_cnt, (head[0], head[1]), False) >= 0:
                                done = True
                                print "Head! Boom!"
                

                        if vary_border_cnt>0 and cv2.pointPolygonTest(self.bounding_cnt, (tail[0], tail[1]), False) >= 0:
                                done = True
                                print "Tail! Boom!"

                elif agent_mode == 1:
                        head_right = [0, 0]
                        head_left = [0, 0]
                        tail_right = [0, 0]
                        tail_left = [0, 0]
                        points = [[]]
                        angle_vertical = angle_in_rad - math.pi/2
                        head_right[0] = int(round(head[0] + np.cos(angle_vertical)*car_body_width))
                        head_right[1] = int(round(head[1] - np.sin(angle_vertical)*car_body_width))
                        head_left[0] = int(head[0]*2 - head_right[0])
                        head_left[1] = int(head[1]*2 - head_right[1])

                        tail_right[0] = int(round(tail[0] + np.cos(angle_vertical)*car_body_width))
                        tail_right[1] = int(round(tail[1] - np.sin(angle_vertical)*car_body_width))
                        tail_left[0] = int(tail[0]*2 - tail_right[0])
                        tail_left[1] = int(tail[1]*2 - tail_right[1])
                        points[0] = (head_left)
                        points.append(head_right)
                        points.append(tail_right)
                        points.append(tail_left)
                        # print points

                        #pygame.draw.rect(self.screen, [255,0,0], ((int(head_left[0]), int(head_left[1])), (car_body_lenth, car_body_width)), 1)
                        for i in range(4):
                                pygame.draw.line(self.screen, [255,255,255], points[0+i], points[(1+i)%4], 1)
                        # pygame.draw.polygon(self.screen, [0,0,0], points)
                        # pygame.draw.circle(self.screen, [255,0,0], (int(head[0]),int(head[1])), 1)

                        if target_pt_mode == 1: # draw
                                pygame.draw.circle(self.screen, [255,0,0], (int(self.car_target_pt[0]),int(self.car_target_pt[1])), 2)


                       
                        pygame.display.update()

                        if head_right[0] <= self.w and head_right[0] >= 0 and head_right[1] <= self.h and head_right[1] >= 0 and tail_right[0] <= self.w and tail_right[0] >= 0 and tail_right[1] <= self.h and tail_right[1] >=0 \
                            and head_left[0] <= self.w and head_left[0] >= 0 and head_left[1] <= self.h and head_left[1] >= 0 and tail_left[0] <= self.w and tail_left[0] >= 0 and tail_left[1] <= self.h and tail_left[1] >=0:
                                done = False
                        else:
                                done = True
                                print "out of playground."

                        if vary_border_cnt>0 and  (cv2.pointPolygonTest(self.bounding_cnt, (head_left[0], head_left[1]), False) >= 0 or cv2.pointPolygonTest(self.bounding_cnt, (head_right[0], head_right[1]), False) >= 0):
                                done = True
                                print "Head! Boom!"
                

                        if vary_border_cnt>0 and (cv2.pointPolygonTest(self.bounding_cnt, (tail_left[0], tail_left[1]), False) >= 0 or cv2.pointPolygonTest(self.bounding_cnt, (tail_right[0], tail_right[1]), False) >= 0):
                                done = True
                                print "Tail! Boom!"

                return done

        def solve_distance_vec(self, distance_vec, intersect_points_vec, center, line_segament, j):
                #print distance_vec
                #time.sleep(1)

                direction_1 = vector_direction(center, line_segament[0])
                direction_2 = vector_direction(center, line_segament[1])
                index_1 = int(- direction_1 +180)
                index_2 = int(- direction_2 + 180)

                if index_1 > index_2:
                        tmp = index_2
                        index_2 = index_1
                        index_1 = tmp

                if index_2 - index_1 > 180:
                        tmp = index_2
                        index_2 = index_1
                        index_1 = tmp

                solving_lenth = (index_2 + 360 - index_1 +1)%360

                solution = np.zeros((360, 2))
                bounding_lines = self.bounding_lines

                k_arr = np.zeros(360)
                line_param_arr = np.ones((360, 2))
                c = np.zeros(360)
                for i in range(360):
                        if i != 180 and i != 0:
                                k_arr[i] = math.tan((i-180)*(math.pi)/180)
                                line_param_arr[i] = (1.00, -1/k_arr[i])
                                c[i] = np.dot(center, line_param_arr[i].T)
                        else:
                                k_arr[i] = 0
                                line_param_arr[i] = (0.00, 1.00)
                                c[i] = np.dot(center, line_param_arr[i].T)

                        # print "bounding_lines len:", len(bounding_lines)
                #print "line_para", line_param_arr[index_1:]
                #print "center", center
                #print "c", c[index_1:]
                #time.sleep(1)

                if index_2 > index_1:
                                for i in range(index_1, index_2):
                                        param = np.zeros((2, 2))
                                        param[0] = line_param_arr[i]
                                        param[1] = (bounding_lines[j][0], bounding_lines[j][1])
                                        bias = (c[i], -bounding_lines[j][2])
                                        if param[0][0]*param[1][1] != param[0][1] * param[1][0]:
                                                solution[i] = np.linalg.solve(param, bias)
                                        #print "direct_1", vector_direction(line_segament[0], solution[i]), "derect_2", vector_direction(line_segament[1], solution[i])
                                        #time.sleep(0.05)
                                        if int(vector_direction(center, solution[i])) == (i-180)*(-1) and abs(vector_direction(line_segament[0], solution[i]) - vector_direction(line_segament[1], solution[i])) == 180:
                                                        if two_point_distance(center, solution[i]) < distance_vec[i]:
                                                                intersect_points_vec[i] = solution[i]
                                                                distance_vec[i] = two_point_distance(center, intersect_points_vec[i])
                                #print index_1, index_2, solution[index_1:], distance_vec[index_1:]

                else:
                                for i in range(index_1, 360):
                                        param = np.zeros((2, 2))
                                        param[0] = line_param_arr[i]
                                        param[1] = (bounding_lines[j][0], bounding_lines[j][1])
                                        bias = (c[i], -bounding_lines[j][2])
                                        if param[0][0]*param[1][1] != param[0][1] * param[1][0]:
                                                solution[i] = np.linalg.solve(param, bias)

                                        #print "direct_1", vector_direction(line_segament[0], solution[i]), "derect_2", vector_direction(line_segament[1], solution[i])
                                        #time.sleep(0.05)
                                        if int(vector_direction(center, solution[i])) == (i-180)*(-1) and abs(vector_direction(line_segament[0], solution[i]) - vector_direction(line_segament[1], solution[i])) == 180:
                                                        if two_point_distance(center, solution[i]) < distance_vec[i]:
                                                                intersect_points_vec[i] = solution[i]
                                                                distance_vec[i] = two_point_distance(center, intersect_points_vec[i])

                                for i in range(0, index_2):
                                        param = np.zeros((2, 2))
                                        param[0] = line_param_arr[i]
                                        param[1] = (bounding_lines[j][0], bounding_lines[j][1])
                                        bias = (c[i], -bounding_lines[j][2])
                                        if param[0][0]*param[1][1] != param[0][1] * param[1][0]:
                                                solution[i] = np.linalg.solve(param, bias)
                                        
                                        if int(vector_direction(center, solution[i])) == (i-180)*(-1) and abs(vector_direction(line_segament[0], solution[i]) - vector_direction(line_segament[1], solution[i])) == 180:
                                                        if two_point_distance(center, solution[i]) < distance_vec[i]:
                                                                intersect_points_vec[i] = solution[i]
                                                                distance_vec[i] = two_point_distance(center, intersect_points_vec[i])

                #print index_1, index_2, solution[index_1:], distance_vec[index_1:]
                #time.sleep(0.5)
                return distance_vec, intersect_points_vec







def two_point_distance(start_pt, end_pt):
        return math.sqrt((start_pt[0]-end_pt[0])**2 + (start_pt[1]-end_pt[1])**2)

def vector_direction(start_pt, end_pt):
        start_pt = axis_convert2_normal(start_pt)
        end_pt = axis_convert2_normal(end_pt)
        pi = math.pi
        angle_in_rad = math.atan2((end_pt[1] - start_pt[1]), (end_pt[0] - start_pt[0]))
        angle_in_degree = (angle_in_rad/math.pi)*180
        return round(angle_in_degree)

def axis_convert2_normal(point_xy_in_video):
        return (point_xy_in_video[0], -point_xy_in_video[1])