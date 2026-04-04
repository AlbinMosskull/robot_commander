#!/usr/bin/env/python3
# File name   : Move.py
# Website     : www.Adeept.com
# Author      : Adeept
# Date        : 2026/01/14
import time
import os
import threading
from board import SCL, SDA
import busio
from adafruit_motor import servo
from adafruit_pca9685 import PCA9685
import Kalman_Filter as Kalman_filter
from mpu6050 import mpu6050
import json
import copy


def load_json(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config
    except (FileNotFoundError, KeyError):
        print("Error reading initial configuration angles")
        return [90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90]  


def save_json(config_data, config_path):
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)
    except Exception:
        print("save config data failed")
        pass

def get_cur_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Directory of servo angle configuration: {script_dir}")
    return script_dir

#             ultra
#                  ________
#     left_I  1 0 |        | 10 11  right_I
#                 |        |
#    left_II  3 2 |  body  | 8 9  right_II
#                 |        |
#   left_III  5 4 |________| 6 7  right_III
class RaspClaws(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(RaspClaws, self).__init__(*args, **kwargs)
        self.__flag = threading.Event()
        self.__flag.clear()
        self.pending_flag = False

        self.i2c = busio.I2C(SCL, SDA)
        self.pca = PCA9685(self.i2c, address=0x5F)
        self.pca.frequency = 50  
        self.servos = [servo.Servo(self.pca.channels[i], min_pulse=500, max_pulse=2400,actuation_range=180) for i in range(16)]
        
        self.servo_json_path = get_cur_path() + "/Servos_Init.json"
        self.init_angles= load_json(self.servo_json_path)["servo_angles"]
        self.last_angles= copy.deepcopy(self.init_angles)
        self.limit_angles_min = [(self.init_angles[i] - 75) for i in range(16)]
        self.limit_angles_max = [(self.init_angles[i] + 75) for i in range(16)]

        self.steady_range_Min = -60
        self.steady_range_Max = 60
        self.error_threshold = 0.1
        self.range_Mid = (self.steady_range_Min + self.steady_range_Max) / 2
        self.X_fix_output = self.range_Mid
        self.Y_fix_output = self.range_Mid
        self.P = 8
        self.scaling_factor = 1.5
        self.target_X = 0
        self.target_Y = 0

        self.kalman_filter_X = Kalman_filter.Kalman_filter(0.001, 0.1)
        self.kalman_filter_Y = Kalman_filter.Kalman_filter(0.001, 0.1)
        
        # MPU6050
        try:
            self.sensor = mpu6050(0x68)
            self.mpu6050_connection = 1
        except:
            self.mpu6050_connection = 0

        self.steadyMode = 0
        self.move_status = False
        self.head_rotate_internal = 0.05
        self.rotate_internal = 0
        # self.rotate_internal = 0.01
        self.height_change = 10
        self.step_internal = 0.13
        # self.step_internal = 0
        self.step_wiggle = 15
        self.direction_command = 'no'
        self.init_all()
        
    def pause(self):
        print('......................pause..........................')
        self.__flag.clear()

    def resume(self):
        print('......................resume......................')
        self.__flag.set()      

    def adjust_angle_track_color(self, servo_index, dir, offset = 2):
        angle_temp = max(self.limit_angles_min[servo_index], min(self.limit_angles_max[servo_index], self.last_angles[servo_index] + offset*dir))
        self.set_servo_angle(servo_index, angle_temp)

    def adjust_init_angle(self, servo_index, dir, offset = 2):
        angle_temp = max(self.limit_angles_min[servo_index], min(self.limit_angles_max[servo_index], self.init_angles[servo_index] + offset*dir))
        self.set_servo_angle(servo_index, angle_temp)
        self.init_angles[servo_index] = angle_temp

    def persist_Servos_init(self , servo_index):
        json_data = load_json(self.servo_json_path)
        json_data["servo_angles"][servo_index] = self.init_angles[servo_index]
        save_json(json_data, self.servo_json_path)

    def init_single_servo(self, servo_index):
        self.set_servo_angle(servo_index, 90)
        self.init_angles[servo_index] = 90

    def set_servo_angle(self, channel, angle):
        # print(f"angle: {angle}")
        angle = max(0, min(180, angle))
        if 0 <= channel < 16:
            self.servos[channel].angle = angle
            time.sleep(self.rotate_internal)
            self.last_angles[channel] = angle

    def body_reset(self):
        self.last_angles= copy.deepcopy(self.init_angles)
        for i in range(12):
            self.servos[i].angle = self.init_angles[i]
            self.last_angles[i] = self.init_angles[i]

    def init_all(self):
        self.last_angles= copy.deepcopy(self.init_angles)
        for i in range(16):
            self.servos[i].angle = self.init_angles[i]

    def ctrl_range(self, raw, max_genout, min_genout):
        return round(max(min_genout, min(max_genout, raw)))

    #left_I right_II left_III
    def first_group_legs_up_down(self, is_up):
        if is_up:
            self.set_servo_angle(1, self.last_angles[1] + 2*self.height_change)
            self.set_servo_angle(9, self.last_angles[9] - 2*self.height_change)
            self.set_servo_angle(5, self.last_angles[5] + 2*self.height_change)            
        else:
            self.set_servo_angle(1, self.last_angles[1] - 2*self.height_change)
            self.set_servo_angle(9, self.last_angles[9] + 2*self.height_change)
            self.set_servo_angle(5, self.last_angles[5] - 2*self.height_change)   
        time.sleep(self.step_internal/3) 

    #right_I left_II right_III
    def second_group_legs_up_down(self, is_up):
        if is_up:
            self.set_servo_angle(11, self.last_angles[11] - 2*self.height_change)
            self.set_servo_angle(3, self.last_angles[3] + 2*self.height_change)
            self.set_servo_angle(7, self.last_angles[7] - 2*self.height_change)
        else:    
            self.set_servo_angle(11, self.last_angles[11] + 2*self.height_change)
            self.set_servo_angle(3, self.last_angles[3] - 2*self.height_change)
            self.set_servo_angle(7, self.last_angles[7] + 2*self.height_change)
        time.sleep(self.step_internal/3)

    #left_I right_II left_III
    def first_group_legs_swing(self, dir, range):
        if "front" == dir:
            self.set_servo_angle(0, self.last_angles[0] + range)
            self.set_servo_angle(4, self.last_angles[4] + range)
            self.set_servo_angle(8, self.last_angles[8] - range)
        elif "back" == dir:
            self.set_servo_angle(0, self.last_angles[0] - range)
            self.set_servo_angle(4, self.last_angles[4] - range)
            self.set_servo_angle(8, self.last_angles[8] + range)         
        elif "left" == dir:
            self.set_servo_angle(0, self.last_angles[0] - range)
            self.set_servo_angle(4, self.last_angles[4] - range)
            self.set_servo_angle(8, self.last_angles[8] - range)
        elif "right" == dir:
            self.set_servo_angle(0, self.last_angles[0] + range)
            self.set_servo_angle(4, self.last_angles[4] + range)
            self.set_servo_angle(8, self.last_angles[8] + range)   
        time.sleep(self.step_internal)        

    #right_I left_II right_III
    def second_group_legs_swing(self, dir, range):
        if "front" == dir:
            self.set_servo_angle(10, self.last_angles[10] - range)
            self.set_servo_angle(6, self.last_angles[6] - range)
            self.set_servo_angle(2, self.last_angles[2] + range) 
        elif "back" == dir:
            self.set_servo_angle(10, self.last_angles[10] + range)
            self.set_servo_angle(6, self.last_angles[6] +  range)
            self.set_servo_angle(2, self.last_angles[2] - range)        
        elif "left" == dir:
            self.set_servo_angle(10, self.last_angles[10] - range)
            self.set_servo_angle(6, self.last_angles[6] - range)
            self.set_servo_angle(2, self.last_angles[2] - range) 
        elif "right" == dir:
            self.set_servo_angle(10, self.last_angles[10] + range)
            self.set_servo_angle(6, self.last_angles[6] +  range)
            self.set_servo_angle(2, self.last_angles[2] + range)    
        time.sleep(self.step_internal)           
           

    def move_forward(self):
        if not self.move_status:
            self.body_reset()
            self.move_status = True
            self.first_group_legs_up_down(True)
            # left_I right_II left_III swing forward
            self.first_group_legs_swing("front", self.step_wiggle)
            # right_I left_II right_III swing backward
            self.second_group_legs_swing("back", self.step_wiggle)      
        else:
            self.first_group_legs_up_down(True)
            # left_I right_II left_III swing forward    
            self.first_group_legs_swing("front", 2*self.step_wiggle)        
            # right_I left_II right_III swing backward
            self.second_group_legs_swing("back", 2*self.step_wiggle)
        
        # left_I right_II left_III  down
        self.first_group_legs_up_down(False)
        
        # right_I left_II right_III up
        self.second_group_legs_up_down(True)
        # right_I left_II right_III swing forward
        self.second_group_legs_swing("front", 2*self.step_wiggle)   
        # left_I right_II left_III swing backward
        self.first_group_legs_swing("back", 2*self.step_wiggle)  

        # right_I left_II right_III down
        self.second_group_legs_up_down(False)


    def move_backward(self):
        if not self.move_status:
            self.body_reset()
            self.move_status = True
            self.first_group_legs_up_down(True)
            self.first_group_legs_swing("back", self.step_wiggle)
            self.second_group_legs_swing("front", self.step_wiggle)          
        else:            
            self.first_group_legs_up_down(True)
            self.first_group_legs_swing("back", 2*self.step_wiggle)
            self.second_group_legs_swing("front", 2*self.step_wiggle)        

        self.first_group_legs_up_down(False)
        
        self.second_group_legs_up_down(True)
        self.second_group_legs_swing("back", 2*self.step_wiggle)  
        self.first_group_legs_swing("front", 2*self.step_wiggle)

        self.second_group_legs_up_down(False)

    def move_left(self):
        if not self.move_status:
            self.body_reset()
            self.move_status = True   
            self.first_group_legs_up_down(True)      
            self.first_group_legs_swing("left", self.step_wiggle)
            self.second_group_legs_swing("right", self.step_wiggle)             
        else:            
            self.first_group_legs_up_down(True)
            self.first_group_legs_swing("left", 2*self.step_wiggle)
            self.second_group_legs_swing("right", 2*self.step_wiggle)  

        self.first_group_legs_up_down(False)
        
        self.second_group_legs_up_down(True)
        self.second_group_legs_swing("left", 2*self.step_wiggle)  
        self.first_group_legs_swing("right", 2*self.step_wiggle)

        self.second_group_legs_up_down(False)


    def move_right(self):
        if not self.move_status:
            self.body_reset()
            self.move_status = True  
            self.first_group_legs_up_down(True)
            self.first_group_legs_swing("right", self.step_wiggle)
            self.second_group_legs_swing("left", self.step_wiggle)            
        else:
            self.first_group_legs_up_down(True)
            self.first_group_legs_swing("right", 2*self.step_wiggle)
            self.second_group_legs_swing("left", 2*self.step_wiggle)  

        self.first_group_legs_up_down(False)
        
        self.second_group_legs_up_down(True)
        self.second_group_legs_swing("right", 2*self.step_wiggle)  
        self.first_group_legs_swing("left", 2*self.step_wiggle)

        self.second_group_legs_up_down(False)

    def stay_steady(self, left_I_input, left_II_input, left_III_input, right_III_input, right_II_input, right_I_input):
        self.set_servo_angle(1, self.init_angles[1] + left_I_input)
        self.set_servo_angle(3, self.init_angles[3] + left_II_input)
        self.set_servo_angle(5, self.init_angles[5] + left_III_input)

        self.set_servo_angle(7, self.init_angles[7] - right_III_input)
        self.set_servo_angle(9, self.init_angles[9] - right_II_input)
        self.set_servo_angle(11, self.init_angles[11] - right_I_input)

    # -------------------------- Self-Balance --------------------------
    def steady(self):
        if not self.mpu6050_connection:
            return
        accelerometer_data = self.sensor.get_accel_data()
        X = accelerometer_data['x']
        Y = accelerometer_data['y']
        X = self.kalman_filter_X.kalman(X)        
        Y = self.kalman_filter_Y.kalman(Y)

        print(f" X: {X} Y: {Y}")
        X_error = self.target_X - X
        Y_error = self.target_Y - Y

        X_error = 0 if abs(X_error) < self.error_threshold else X_error
        Y_error = 0 if abs(Y_error) < self.error_threshold else Y_error

        if X_error == 0 and Y_error == 0:
            return

        self.X_fix_output = X_error * self.P
        self.Y_fix_output = Y_error * self.P

        X_ratio = abs(self.X_fix_output)/((abs(self.X_fix_output)+abs(self.Y_fix_output)))
        Y_ratio = 1-X_ratio

        print(f"PID  X_fix_output: {self.X_fix_output} Y_fix_output: {self.Y_fix_output}")
        
        if Y_ratio > X_ratio:
            if Y <= 0:
                left_I_input = self.ctrl_range((self.Y_fix_output*Y_ratio -self.X_fix_output*X_ratio)/self.scaling_factor, self.steady_range_Max, self.steady_range_Min)
                right_III_input = self.ctrl_range((-self.Y_fix_output*Y_ratio + self.X_fix_output*X_ratio)/self.scaling_factor, self.steady_range_Max, self.steady_range_Min)
                left_III_input = self.ctrl_range((-self.Y_fix_output*Y_ratio -self.X_fix_output*X_ratio)*self.scaling_factor, self.steady_range_Max, self.steady_range_Min)
                right_I_input = self.ctrl_range((self.Y_fix_output*Y_ratio + self.X_fix_output*X_ratio)*self.scaling_factor, self.steady_range_Max, self.steady_range_Min)
            else:
                left_I_input = self.ctrl_range((self.Y_fix_output*Y_ratio -self.X_fix_output*X_ratio)*self.scaling_factor, self.steady_range_Max, self.steady_range_Min)
                right_III_input = self.ctrl_range((-self.Y_fix_output*Y_ratio + self.X_fix_output*X_ratio)*self.scaling_factor, self.steady_range_Max, self.steady_range_Min)
                left_III_input = self.ctrl_range((-self.Y_fix_output*Y_ratio -self.X_fix_output*X_ratio)/self.scaling_factor, self.steady_range_Max, self.steady_range_Min)
                right_I_input = self.ctrl_range((self.Y_fix_output*Y_ratio + self.X_fix_output*X_ratio)/self.scaling_factor, self.steady_range_Max, self.steady_range_Min)
            left_II_input = self.ctrl_range((self.Y_fix_output-self.X_fix_output)*X_ratio/2, self.steady_range_Max, self.steady_range_Min)
            right_II_input = self.ctrl_range((self.X_fix_output-self.Y_fix_output)*X_ratio/2, self.steady_range_Max, self.steady_range_Min)                    
        else:
            if X > 0:
                left_I_input = self.ctrl_range((self.Y_fix_output*Y_ratio -self.X_fix_output*X_ratio)/self.scaling_factor, self.steady_range_Max, self.steady_range_Min)
                left_II_input = self.ctrl_range((self.Y_fix_output-self.X_fix_output)*X_ratio/2/self.scaling_factor, self.steady_range_Max, self.steady_range_Min)              
                left_III_input = self.ctrl_range((-self.Y_fix_output*Y_ratio -self.X_fix_output*X_ratio)/self.scaling_factor, self.steady_range_Max, self.steady_range_Min)
                right_III_input = self.ctrl_range((-self.Y_fix_output*Y_ratio + self.X_fix_output*X_ratio)*self.scaling_factor, self.steady_range_Max, self.steady_range_Min)
                right_II_input = self.ctrl_range((self.X_fix_output-self.Y_fix_output)*X_ratio/2*self.scaling_factor, self.steady_range_Max, self.steady_range_Min)
                right_I_input = self.ctrl_range((self.Y_fix_output*Y_ratio + self.X_fix_output*X_ratio)*self.scaling_factor, self.steady_range_Max, self.steady_range_Min)
            else:
                left_I_input = self.ctrl_range((self.Y_fix_output*Y_ratio -self.X_fix_output*X_ratio)*self.scaling_factor, self.steady_range_Max, self.steady_range_Min)
                left_II_input = self.ctrl_range((self.Y_fix_output-self.X_fix_output)*X_ratio/2*self.scaling_factor, self.steady_range_Max, self.steady_range_Min)               
                left_III_input = self.ctrl_range((-self.Y_fix_output*Y_ratio -self.X_fix_output*X_ratio)*self.scaling_factor, self.steady_range_Max, self.steady_range_Min)
                right_III_input = self.ctrl_range((-self.Y_fix_output*Y_ratio + self.X_fix_output*X_ratio)/self.scaling_factor, self.steady_range_Max, self.steady_range_Min)
                right_II_input = self.ctrl_range((self.X_fix_output-self.Y_fix_output)*X_ratio/2/self.scaling_factor, self.steady_range_Max, self.steady_range_Min)
                right_I_input = self.ctrl_range((self.Y_fix_output*Y_ratio + self.X_fix_output*X_ratio)/self.scaling_factor, self.steady_range_Max, self.steady_range_Min)


        # left_I_input = self.ctrl_range(self.Y_fix_output*Y_ratio -self.X_fix_output*X_ratio, self.steady_range_Max, self.steady_range_Min)

        # right_I_input = self.ctrl_range(self.Y_fix_output*Y_ratio + self.X_fix_output*X_ratio, self.steady_range_Max, self.steady_range_Min)

        # left_II_input = self.ctrl_range((self.Y_fix_output-self.X_fix_output)*X_ratio, self.steady_range_Max, self.steady_range_Min)

        # right_II_input = self.ctrl_range((self.X_fix_output-self.Y_fix_output)*X_ratio, self.steady_range_Max, self.steady_range_Min)
 
        # left_III_input = self.ctrl_range(-self.Y_fix_output*Y_ratio -self.X_fix_output*X_ratio, self.steady_range_Max, self.steady_range_Min)

        # right_III_input = self.ctrl_range(-self.Y_fix_output*Y_ratio + self.X_fix_output*X_ratio, self.steady_range_Max, self.steady_range_Min)

        self.stay_steady(left_I_input, left_II_input, left_III_input, right_III_input, right_II_input, right_I_input)
        time.sleep(0.05)

    def run(self):
        while True:
            self.__flag.wait() 
            if not self.steadyMode:
                if self.direction_command == 'forward':
                    self.move_forward()
                elif self.direction_command == 'backward':
                    self.move_backward()
                elif self.direction_command == 'left': 
                    self.move_left()
                elif self.direction_command == 'right':
                    self.move_right()
                elif self.direction_command == 'stand':
                    self.body_reset()
                elif self.direction_command == 'lookleft': 
                    self.adjust_init_angle(12, 1, 1)
                    time.sleep(self.head_rotate_internal)
                elif self.direction_command == 'lookright':
                    self.adjust_init_angle(12,-1, 1)
                    time.sleep(self.head_rotate_internal)
                elif self.direction_command == 'up': 
                    self.adjust_init_angle(13, -1, 1)
                    time.sleep(self.head_rotate_internal)
                elif self.direction_command == 'down':
                    self.adjust_init_angle(13, 1, 1)
                    time.sleep(self.head_rotate_internal)
            else:
                self.steady()

            if self.pending_flag:
                time.sleep(0.2)
                self.pause()
                self.body_reset()
                
            

    def command_input(self, command):
        self.pending_flag = False
        if command == 'steadyCamera':
            self.steadyMode = 1
            self.resume()
        elif command == 'steadyCameraOff':
            self.pending_flag = True
            self.steadyMode = 0

        if self.steadyMode == 0:
            if command == "slow":
                # self.rotate_internal = 0.03
                self.step_internal = 0.04
            elif command == "fast":
                # self.rotate_internal = 0
                self.step_internal = 0.13
            elif command == "stand":
                self.direction_command = 'stand'
                time.sleep(0.3)
                self.pending_flag = True
                self.move_status = False
                self.direction_command = 'no'                
            elif command != 'get_info':
                if self.direction_command != command:
                    self.move_status = False
                    self.direction_command = command
                    self.resume()

    def cleanup(self):
        self.pca.deinit() 

if __name__ == '__main__':
    robot = RaspClaws()
    robot.start()

    # robot.command_input('forward')
    # time.sleep(5)
    # robot.command_input('stand')
    # robot.command_input('backward')
    # time.sleep(5)
    # robot.command_input('stand')
    # robot.command_input('left')
    # time.sleep(5)
    # robot.command_input('stand')
    # robot.command_input('right')
    # time.sleep(5)  

    # robot.command_input('stand')
    # robot.command_input("steadyCamera")
    count = 0
    while count < 90:
        robot.adjust_angle_track_color(12, 1)
        time.sleep(0.05)
        count += 1
    while count > 0:
        robot.adjust_angle_track_color(12, -1)
        time.sleep(0.05)
        count -= 1 
        

    try:
        while True:
            # robot.adjust_angle_track_color(12, -1)
            time.sleep(0.05)
          
            # robot.move_forward()
            # robot.move_backward()
            # robot.move_left()
            # robot.move_right()

            # robot.command_input('forward')
            # time.sleep(5)
            # robot.move_stop()
            
            # robot.command_input('backward')
            # time.sleep(5)
            # robot.move_stop()

            # robot.command_input('left')
            # time.sleep(5)
            # robot.move_stop()

            # robot.command_input('right')
            # time.sleep(5)  
            # robot.move_stop()      
            # robot.command_input('no')
    except KeyboardInterrupt:
        robot.body_reset()
        robot.pause()

