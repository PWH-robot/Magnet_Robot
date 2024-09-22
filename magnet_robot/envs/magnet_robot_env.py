import os
import gym
from gym import spaces
import numpy as np
import math
import cv2
import pylab
import os
import time
from dynamixel_sdk import *
import pandas as pd

if os.name == 'nt':
    import msvcrt
    def getch():
        return msvcrt.getch().decode()

ADDR_PRO_TORQUE_ENABLE      = 64
ADDR_PRO_PROFILE_VELOCITY   = 112
ADDR_PRO_GOAL_POSITION      = 116
ADDR_PRO_PRESENT_POSITION   = 132

PROTOCOL_VERSION            = 2.0

DXL_ID                      = 1
BAUDRATE                    = 57600
DEVICENAME                  = 'COM18' 

TORQUE_ENABLE               = 1
TORQUE_DISABLE              = 0                  
DXL_MOVING_STATUS_THRESHOLD = 20               

portHandler = PortHandler(DEVICENAME)

packetHandler = PacketHandler(PROTOCOL_VERSION)

if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    print("Press any key to terminate...")
    getch()
    quit()

if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    print("Press any key to terminate...")
    getch()
    quit()

dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_PRO_TORQUE_ENABLE, TORQUE_ENABLE)
if dxl_comm_result != COMM_SUCCESS:
    print("%s" % packetHandler.getTxRxResult(dxl_comm_result))
elif dxl_error != 0:
    print("%s" % packetHandler.getRxPacketError(dxl_error))
else:
    print("Dynamixel has been successfully connected")

dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_PRO_PROFILE_VELOCITY, 100)


scores, episodes, times, robot_height, magnet_height = [], [], [], [], []

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

cap = cv2.VideoCapture(1)

class Magnet_robot(gym.Env):

    def __init__(self):
        self.action_space = spaces.Box(low=-1.5, high=1.5, shape=(1, ), dtype="float32")
        self.observation_space = spaces.Box(low=0, high=50**2, shape=(2,), dtype="float32")
        self.done = False
        self.episode = 0
        self.train = False
        self.tim = time.time()
        self.total_return = 0
        self.score_avg = 0
        self.target_base = 375

    def step(self, action):
        if action[0] < -1:
            action[0] = -1
        elif action[0] > 1:
            action[0] = 1
        goal_position = round(action[0] * 270) + 1030
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_PRO_GOAL_POSITION, goal_position)
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRO_PRESENT_POSITION)
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 7, (255, 0, 0), -1)
                cv2.putText(frame, f"Center: {cx}, {cy}", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow('Red Object Detection', frame)
        state = []
        state.append(cy)
        state.append(dxl_present_position)
        reward = 6 * math.exp(-0.005*((self.target_base - cy)**2)) - 3
        if self.train == False:
            magnet_heigh = (abs(cy - 400)/60) * 1.3 + 0.5
            angle = (dxl_present_position/4096) * 2 * 3.1415926535
            if math.cos(angle) >= 0:
                robot_heigh = abs(3 * math.cos(angle) - 1.18)
            elif math.cos(angle) < 0:
                robot_heigh = -3 * math.cos(angle) + 1.18
            self.plot_height(magnet_heigh, robot_heigh)
        if (time.time() - self.tim >= 10) and (self.train == True):
            self.done = True
            self.plot()
            self.episode = self.episode + 1
        self.total_return  = self.total_return + reward
        return state, reward, self.done, {}
    
    def settings(self, rend, train):
        self.train = train

    def reset(self):
        self.tim = time.time()
        self.total_return = 0
        self.done = False
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_PRO_PRESENT_POSITION)
        dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_PRO_GOAL_POSITION, dxl_present_position)
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cx, cy), 7, (255, 0, 0), -1)
                cv2.putText(frame, f"Center: {cx}, {cy}", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow('Red Object Detection', frame)
        state = []
        state.append(cy)
        state.append(dxl_present_position)
        return state

    def plot(self): #plot score graph
        self.score_avg = 0.9 * self.score_avg + 0.1 * self.total_return if self.episode != 0 else self.total_return 
        scores.append(self.score_avg)
        episodes.append(self.episode)
        pylab.plot(episodes, scores, 'b')
        pylab.xlabel("episode")
        pylab.ylabel("average score")
        pylab.savefig("PPO_reward.png")

    def plot_height(self, magnet_heigh, robot_heigh): #plot score graph
        times.append(time.time() - self.tim)
        magnet_height.append(magnet_heigh)
        robot_height.append(robot_heigh)

    def save_data(self):
        index = ['Time', 'Magnet_height', 'Robot_height']
        data = [times, magnet_height, robot_height]
        df = pd.DataFrame(data, index=index).T
        df.to_excel('gwanhubabo.xlsx', index=False)

