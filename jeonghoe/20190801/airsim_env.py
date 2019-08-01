import setup_path
import math
import numpy as np
from numpy import linalg as LA
from airsim_base_env import AirSimBaseEnv

way_point_unit = 10  # meter


class AirSimEnv(AirSimBaseEnv):

    @staticmethod
    def get_state_size():
        return 14

    def get_current_state(self, car_state, car_prev_state, way_points, check_point, all_obstacles):

        state = []

        # ======
        # (1) Forward angle
        # ======
        # 현재 주행 구간에서 2 개 각도 (max 10개)
        forward_angle_arr = self.get_track_forward_angle(car_state, way_points, check_point)
        
        for i in range(3):
            if i == 0:
                continue
            forward_angle = round(abs(forward_angle_arr[i]), 2)

            if forward_angle_arr[i] < 0:
                forward_angle = forward_angle * -1

            state.append(forward_angle)
        '''
        sign = 0
        for i in range(2):
            forward_angle = round(abs(forward_angle_arr[i]), 2)

            if forward_angle_arr[i] < 0:
                forward_angle = forward_angle * -1

            if forward_angle > 0:
                if sign == 0:
                    sign = 1
                elif sign != 1:
                    sign = 3
            elif forward_angle < 0:
                if sign == 0:
                    sign = 2
                elif sign != 2:
                    sign = 3

        state.append(sign)
        '''
        # ======
        # (2) Moving angle
        # ======
        angle = self.get_moving_angle(car_prev_state, car_state, way_points, check_point)
        for i in range(3):
            if i == 0:
                continue
            f_angle = round(abs(forward_angle_arr[i]), 2)
            if forward_angle_arr[i] < 0:
                f_angle = f_angle * -1
            state.append(angle - f_angle)

        # ======
        # (3) Current distance from center(position)
        # ======
        dist = round(self.get_distance_from_center(car_state, way_points, check_point), 2)
        if self.is_right_of_center(car_state, way_points, check_point):
            state.append(dist)
        else:
            state.append(dist * -1)
        
        # ======
        # (4) Current velocity
        # ======
        velocity = self.get_speed(car_state)
        state.append(velocity)

        # ======
        # (5, 6) Obstacle distance, to middle
        # ======
        maxObstacleSize = 4
        track_forward_obstacles = self.get_track_forward_obstacle(car_state, way_points, check_point, all_obstacles)
        if len(track_forward_obstacles) > 0:
            for i in range(len(track_forward_obstacles)):
                o_dist, o_to_middle = track_forward_obstacles[i]

                if o_dist>0 and o_dist < 40:
                    state.append(round(o_dist, 1))
                    state.append(round(o_to_middle, 2))
                    maxObstacleSize -= 1


        if maxObstacleSize>0:
            for i in range(maxObstacleSize):
                state.append(0)
                state.append(0)
        #print("[{},{},{},{}], [{}], [{}], [], Moving Angle: {}".format(state[0],state[1],state[2],state[3],state[4],state[5], state[6]))
        return state
