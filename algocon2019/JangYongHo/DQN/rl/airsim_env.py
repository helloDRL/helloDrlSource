import setup_path
import math
import numpy as np
from numpy import linalg as LA

way_point_unit = 10  # meter


class AirSimEnv:

    @staticmethod
    def get_state_size():
        return 6

    def get_current_state(self,car_state, car_prev_state, way_points, check_point, all_obstacles):

        state = []

        #======
        # (1) Forward angle
        # ======
        #현재 주행 구간에서 10 개 각도
        forward_angle_arr = self.get_track_forward_angle(car_state, way_points, check_point)

        forward_angle = round(abs(forward_angle_arr[2]),2)

        if forward_angle_arr[2] < 0 :
            forward_angle = forward_angle * -1

        state.append(forward_angle)
        # ======
        # (2) Moving angle
        # ======
        angle = self.get_moving_angle(car_prev_state, car_state, way_points, check_point)

        state.append(angle)

        # ======
        # (3) Current distance from center(position)
        # ======
        dist = round(self.get_distance_from_center(car_state, way_points, check_point),2)

        # road width = 10 m

        if self.is_right_of_center(car_state, way_points, check_point):
            state.append(dist)
        else :
            state.append(dist * -1)

        # ======
        # (4) Current velocity
        # ======
        velocity = self.get_speed(car_state)
        state.append(velocity)



        # ======
        # (5, 6) Obstacle distance, to middle
        # ======
        track_forward_obstacles = self.get_track_forward_obstacle(car_state, way_points,check_point, all_obstacles)
        if len(track_forward_obstacles) > 0:
            o_dist, o_to_middle = track_forward_obstacles[0]
            if o_dist < 50 :
                state.append(round(o_dist, 1))
                state.append(round(o_to_middle,2))
            else :
                state.append(0)
                state.append(0)
        else :
            state.append(0)
            state.append(0)

        return state

    def load_track_info(self, client):
        algo_apis = client.getAlgoUserAPI()
        way_points_raw = algo_apis.wayPoints
        obstacle_points_raw = algo_apis.ac_block_points
        way_points_revised = []
        obstacle_points_revised = []
        # 데이터 정제
        for item in way_points_raw:
            if len(item):  # 데이터 있는것만
                way_points_revised.append(item)
        # print(obstacle_points_raw)
        for item in obstacle_points_raw:
            if len(item):
                obstacle_points_revised.append(item)

        return np.array(way_points_revised), np.array(obstacle_points_revised)

    # ============================
    # 주행 상태 관련
    # get distance from center line
    # ============================
    def get_distance_from_center(self, car_state, way_points, check_point):
        prev, next = self.get_current_way_points(car_state, way_points, check_point)
        pd = car_state.kinematics_estimated.position
        car_pt = np.array([pd.x_val, pd.y_val, 0])
        dist = np.linalg.norm(
            np.cross((car_pt - way_points[prev]), (way_points[next] - way_points[prev]))) / np.linalg.norm(
            way_points[next] - way_points[prev])
        # print("prev_way_num:{}, next_way_num:{}".format(prev, next))
        return dist

    # ============================
    # 자동차가 트랙의 중앙선 왼쪽에 있는지를 알려줍니다.
    # ============================
    def is_right_of_center(self, car_state, way_points, check_point):
        prev, next = self.get_current_way_points(car_state, way_points, check_point)
        pd = car_state.kinematics_estimated.position
        car_pt = np.array([pd.x_val, pd.y_val, 0])
        return self.get_cross_product_element_sign(car_pt, way_points[prev], way_points[next])

    def get_cross_product_element_sign(self, car_pt, wp_1, wp_2):
        v1 = car_pt - wp_1
        v2 = wp_2 - wp_1
        return np.cross(v1, v2)[2] < 0

    # ============================
    # 단순 Full scan 으로 장애물 앞뒤 way point 그리고 way point 앞에서부터 거리, 장애물의 to middle 값을 리턴합니다. 주행중인 자동차에는 도로 이탈이 심하므로 사용못함
    # (binary search 를 사용 못했을때 값 계산이 틀리는 이유는 도로가 곡선이 많이 있기 때문에 두점의 직선거리의 절반 지점이, 인덱스의 절반이 아닌 경우가 많다.)
    # ============================

    def get_current_obstacle_info_full_scan(self, obstacle_point, way_points):
        last_index = len(way_points) - 1
        min_dist = 999999
        min_index = last_index

        for x in range(0, last_index):
            calc_dist = np.linalg.norm(way_points[x] - obstacle_point)
            if calc_dist < min_dist:
                min_dist = calc_dist
                min_index = x

        candidate_prev_index = self.get_next_N_waypoint_index(min_index, -1, way_points)
        candidate_next_index = self.get_next_N_waypoint_index(min_index, +1, way_points)

        prev_dist = LA.norm(obstacle_point - way_points[candidate_prev_index])
        next_dist = LA.norm(obstacle_point - way_points[candidate_next_index])

        # prev_dist 가 크거나 하필 두 거리가 같으면 전방(Goal 방향)의 구간을 택하는것으로 정함
        if prev_dist >= next_dist:
            wp_idx_1 = min_index
            wp_idx_2 = candidate_next_index
        # next_dist 가 크면
        else:
            wp_idx_1 = candidate_prev_index
            wp_idx_2 = min_index

        dist_from_wp1 = self.get_dist_to_intersection_point(obstacle_point, way_points[wp_idx_1], way_points[wp_idx_2])
        dist_from_wp1 = way_point_unit if dist_from_wp1 > way_point_unit else 0 if dist_from_wp1 < 0 else dist_from_wp1
        dist_from_wp1 = round(dist_from_wp1, 2)

        # get_distance_from_center()
        to_middle = np.linalg.norm(
            np.cross((obstacle_point - way_points[wp_idx_1]),
                     (way_points[wp_idx_2] - way_points[wp_idx_1]))) / np.linalg.norm(
            way_points[wp_idx_2] - way_points[wp_idx_1])

        if not self.get_cross_product_element_sign(obstacle_point, way_points[wp_idx_1], way_points[wp_idx_2]):
            to_middle = to_middle * -1

        return wp_idx_1, wp_idx_2, dist_from_wp1, to_middle

    def get_all_obstacle_info(self, obstacles, way_points):
        # print(obstacles)
        obstacle_sectors = []
        for x in obstacles:
            # print(x)
            # obstacle_sectors.append(self.get_current_way_points_binary(x, way_points))
            obstacle_sectors.append(self.get_current_obstacle_info_full_scan(x, way_points))
        # print(obstacle_sectors)
        return obstacle_sectors

    # ============================
    # [input] object(car 또는 obstacle)의 좌표와 그 앞 뒤 좌표
    # 앞, 뒤 좌표를 연결한 직선과 object 좌표점이 직교하는 교차점을 구하여 앞 좌표에서부터 교차점까지의 거리를 리턴한다.
    # 이 때, way_point_unit 은 고정된 값(10m)
    # ============================
    def get_dist_to_intersection_point(self, object_pt, prev_pt, next_pt):
        cosine_val = np.dot(object_pt - prev_pt, next_pt - prev_pt) \
                     / LA.norm(next_pt - prev_pt) / LA.norm(object_pt - prev_pt)
        portion = LA.norm(object_pt - prev_pt) * cosine_val / LA.norm(next_pt - prev_pt)
        dist = way_point_unit * portion
        return way_point_unit if dist > way_point_unit else 0 if dist < 0 else dist

    # ============================
    # 현재 주행중인 차가 속해있는 구간 즉, 자동차 바로 앞과 뒤 way point index 를 리턴합니다.
    # window (-1, 10) 를 설정하고 전수 검사를 수행
    # ============================
    def get_current_way_points(self, car_state, way_points, check_point):
        pd = car_state.kinematics_estimated.position
        car_pt = np.array([pd.x_val, pd.y_val, 0])

        ##윈도우 설정
        if check_point == False:
            first, last = -1, 10
        else:
            first = self.get_next_N_waypoint_index(check_point, -1, way_points)
            last = self.get_next_N_waypoint_index(check_point, 10, way_points)

        # print("window:{},{}".format(first, last))
        max_index = len(way_points) - 1
        min_dist = 100000
        min_dist_idx = 0

        # 정상 케이스
        if first < last:
            for x in range(first, last):
                calc_dist = np.linalg.norm(way_points[x] - car_pt)
                if min_dist > calc_dist:
                    min_dist = calc_dist
                    min_dist_idx = x
        else:
            for x in range(first, max_index):
                calc_dist = np.linalg.norm(way_points[x] - car_pt)
                if min_dist > calc_dist:
                    min_dist = calc_dist
                    min_dist_idx = x

            for x in range(0, last):
                calc_dist = np.linalg.norm(way_points[x] - car_pt)
                if min_dist > calc_dist:
                    min_dist = calc_dist
                    min_dist_idx = x

        # min 값이 차의 전방인지 후방인지 구한다.
        min_dist_prev_idx = self.get_next_N_waypoint_index(min_dist_idx, -1, way_points)
        min_dist_next_idx = self.get_next_N_waypoint_index(min_dist_idx, 1, way_points)

        calc_dist_prev = np.linalg.norm(way_points[min_dist_prev_idx] - car_pt)
        calc_dist_next = np.linalg.norm(way_points[min_dist_next_idx] - car_pt)
        if calc_dist_prev < calc_dist_next:
            return min_dist_prev_idx, min_dist_idx
        else:
            return min_dist_idx, min_dist_next_idx

    # ============================
    # 자동차의 현재 속도를 알려줍니다.
    # ===========================
    def get_speed(self, car_state):
        return round(car_state.speed * 3.6, 2)

    # ============================
    # 진행방향이 앞(True)인가 뒤(False)인가
    # ============================
    def is_moving_forward(self, prev_state, current_state, way_points, check_point):
        prev, next = self.get_current_way_points(current_state, way_points, check_point)
        pd = current_state.kinematics_estimated.position
        car_pt = np.array([pd.x_val, pd.y_val, 0])
        pd = prev_state.kinematics_estimated.position
        prev_car_pt = np.array([pd.x_val, pd.y_val, 0])
        v1 = car_pt - prev_car_pt
        v2 = way_points[next] - way_points[prev]
        if np.dot(v1, v2) == 0 or (LA.norm(v1) * LA.norm(v2)) == 0:
            return True
        check_angle = np.dot(v1, v2) / (LA.norm(v1) * LA.norm(v2))
        if check_angle > 1:
            check_angle = 1
        elif check_angle < -1:
            check_angle = -1
        angle = math.acos(check_angle) * 180 / math.pi
        return -90 < angle < 90

    # ============================
    # 진행방향이 도로 진행방향과 얼마나 align 되어있는가
    # + 이면 왼쪽으로 기울어짐, - 이면 오른쪽으로 기울어짐
    # ex) -30 : 왼쪽으로 30도 기울어진 진행방향이다.
    # 후진 할 때에도 동일한 각을 보여준다.
    # ============================
    def get_moving_angle(self, prev_state, current_state, way_points, check_point):
        prev, next = self.get_current_way_points(current_state, way_points, check_point)
        pd = current_state.kinematics_estimated.position
        car_pt = np.array([pd.x_val, pd.y_val, 0])
        pd = prev_state.kinematics_estimated.position
        prev_car_pt = np.array([pd.x_val, pd.y_val, 0])

        v1 = car_pt - prev_car_pt
        v2 = way_points[next] - way_points[prev]

        # 움직이지 않으면 0 리턴
        if (LA.norm(v1) == 0):
            return 0

        angle = self.get_v_angle(v1, v2)

        if not self.is_moving_forward(prev_state, current_state, way_points, check_point):
            if angle > 0:
                angle = angle - 180
            else:
                angle = angle + 180
        return round(angle, 1)

    def get_v_angle(self, v1, v2):
        check_angle = np.dot(v1, v2) / (LA.norm(v1) * LA.norm(v2))
        if check_angle > 1:
            check_angle = 1
        elif check_angle < -1:
            check_angle = -1
        angle = round(math.acos(check_angle) * 180 / math.pi, 1)
        if np.cross(v1, v2)[2] > 0:
            return angle * -1
        else:
            return angle

    # ============================
    # 전체도록의 몇 퍼센트 주행했지 알려줍니다.
    # default 1바퀴 기준으로 되어있으며, 상태에 따라 값을 받아와서 넣어주어야 함.
    # ============================

    def get_progress(self, car_state, way_points, check_point, cur_lab=1, total_lap=2):
        prev, next = self.get_current_way_points(car_state, way_points, check_point)
        curr = next + (len(way_points)-1) * (cur_lab - 1)
        total = (len(way_points)-1) * total_lap
        returnValue = round((curr/total) * 100 , 2)
        return returnValue
    # ============================
    # 트랙 전방 도로의 각도를 보여준다. 10개 구간
    # ============================
    def get_track_forward_angle(self, car_state, way_points, check_point):
        track_angle = []
        # 일단은 전방 10개 waypoint 기준으로 계산
        prev, next = self.get_current_way_points(car_state, way_points, check_point)
        v1 = way_points[next] - way_points[prev]
        # print("기준 wp: {} ~ {}".format(prev, next))
        for x in range(0, 10):
            t1 = self.get_next_N_waypoint_index(next, x, way_points)
            t2 = self.get_next_N_waypoint_index(next, x + 1, way_points)
            # print("타겟 wp: {} ~ {}".format(t1, t2))
            v2 = way_points[t1] - way_points[t2]
            angle = self.get_v_angle(v1, v2)

            if angle > 0:
                angle = 180 - angle
            else:
                angle = -180 - angle

            track_angle.append(round(angle))
        return track_angle

        # ============================
        # 트랙 전방 도로의 장애물 정보를 보여준다. 10개 구간(100m)
        # ============================

    def get_track_forward_obstacle(self, car_state, way_points, check_point, all_obstacles):
        # 장애물 사이즈
        # 장애물 거리
        # 장애물 놓인 위치
        track_obstacles = []
        # 일단은 전방 10개 waypoint 기준으로 계산
        prev, next = self.get_current_way_points(car_state, way_points, check_point)

        pd = car_state.kinematics_estimated.position
        car_pt = np.array([pd.x_val, pd.y_val, 0])

        car_dist_from_prev = self.get_dist_to_intersection_point(car_pt, way_points[prev], way_points[next])
        car_dist_to_next = way_point_unit - car_dist_from_prev

        # print("기준 wp: {} ~ {}".format(prev, next))
        for x in range(0, 9):  # 9로 변경
            t1 = self.get_next_N_waypoint_index(next, x, way_points)
            t2 = self.get_next_N_waypoint_index(next, x + 1, way_points)
            # print("타겟 wp: {} ~ {}".format(t1, t2))
            for obs in all_obstacles:
                if x == 0 and prev == obs[0] and next == obs[1]:
                    # 같은 way_point 안에 있을 때
                    dist = round(obs[2] - car_dist_from_prev, 5)
                    track_obstacles.append((dist, round(obs[3], 5)))
                elif t1 == obs[0] and t2 == obs[1]:
                    # 다른 way_point 안에 있을 때
                    dist = round(x * way_point_unit + car_dist_to_next + obs[2], 5)
                    track_obstacles.append((dist, round(obs[3], 5)))

        return track_obstacles

    # privates
    def get_next_N_waypoint_index(self, current_index, n, way_points):
        next_index = current_index + n
        max_index = len(way_points) - 1

        if next_index > max_index:
            next_index = next_index - max_index - 1
        elif next_index < 0:
            next_index = next_index + max_index + 1
        return next_index

    def nparray(self, state):
        return np.array([state.x_val, state.y_val, state.z_val])

    def norm(self, state):
        return LA.norm(np.array([state.x_val, state.y_val, state.z_val]))
