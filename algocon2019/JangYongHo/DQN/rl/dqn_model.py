import setup_path
import airsim
import os
import time
import math
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from airsim_env import AirSimEnv
from abc import abstractmethod

# =========================================================== #
# Global Configurations
# =========================================================== #
enable_api_control = True  # True(Api Control) /False(Key board control)
is_debug = False
current_clock_speed = 1

# =========================================================== #


# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, dqn_param):

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = dqn_param.discount_factor
        self.learning_rate = dqn_param.learning_rate
        self.epsilon_decay = dqn_param.epsilon_decay
        self.epsilon_min = dqn_param.epsilon_min
        self.epsilon = dqn_param.epsilon
        self.batch_size = dqn_param.batch_size
        self.train_start = dqn_param.train_start
        # 리플레이 메모리, 최대 크기 20000
        self.memory = deque(maxlen=dqn_param.memory_size)

        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()

    def load_model(self, file_path):
        if not os.path.exists(file_path):
            raise IncorrectAction("Weight file not found!!")
        else:
            self.model.load_weights(file_path)

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(36, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(36, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def get_eval_action(self, state):
        q_value = self.model.predict(state)
        return np.argmax(q_value[0])

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


# DQN Hyper param
class DQNParam:
    # default values
    discount_factor = 0.99
    learning_rate = 0.00025
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01
    batch_size = 100
    train_start = 1000
    # replay memory size : max 20000
    memory_size = 20000


class CarState:
    def __init__(self, name):
        self.__name = name

    collided = False
    collision_distance = 0
    speed = 0
    to_middle = 0
    moving_angle = 0

    moving_forward = True
    lap_progress = 0
    track_forward_angles = []
    track_forward_obstacles = []


class DQNClient:

    def __init__(self, dqn_param):
        self.player_name = ""

        self.car_controls = airsim.CarControls()

        self.client = airsim.CarClient()
        self.client.reset()
        self.client.confirmConnection()
        self.client.enableApiControl(enable_api_control, self.player_name)

        self.airsim_env = AirSimEnv()
        self.way_points, self.obstacle_points = self.airsim_env.load_track_info(self.client)
        self.collision_time_stamp = 0
        self.sensing_info = CarState(self.player_name)
        self.all_obstacles = self.airsim_env.get_all_obstacle_info(self.obstacle_points, self.way_points)
        self.state_size = self.airsim_env.get_state_size()
        self.frozen_count = 0
        self.car_current_pos_x, self.car_next_pos_x = 0, 0

        # road half width + car half width
        self.half_road_limit = self.client.getAlgoUserAPI().ac_road_width_half + 1.25

        self.control_interval = round(0.1 / current_clock_speed,2)

        if len(self.action_space()) < 1:
            raise IncorrectAction("Please check the action definition : At least one action is required")

        self.action_size = len(self.action_space())

        # DQN 에이전트 생성
        self.agent = DQNAgent(self.state_size, self.action_size, dqn_param)

        # running client id 로 폴더 생성
        now = time.localtime()
        self.run_cid = "T%02d%02d_%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        # 시간 생성.
        self.start_time = time.time()
        self.end_time = 0


    @staticmethod
    def make_initial_movement(car_controls, client):
        # 조금 주행을 시킨다.
        car_controls.throttle = 1
        car_controls.steering = 0
        client.setCarControls(car_controls)
        time.sleep(round(2/current_clock_speed,2))

    def calc_sensing_data(self, car_next_state, car_current_state, backed_car_state, way_points, check_point_index):
        distance_from_center = self.airsim_env.get_distance_from_center(car_next_state, way_points,
                                                                        check_point_index)
        right_of_center = self.airsim_env.is_right_of_center(car_next_state, way_points, check_point_index)
        self.sensing_info.to_middle = distance_from_center * (1 if right_of_center else -1)
        self.sensing_info.speed = self.airsim_env.get_speed(car_next_state)
        self.sensing_info.moving_forward = self.airsim_env.is_moving_forward(car_current_state, car_next_state,
                                                                             way_points,
                                                                             check_point_index)
        # 정지해 있는 상태에서 각도를 구할 수 없으므로, 좌표가 달랐던 마지막 상태를 기억하여 둔다.
        self.car_current_pos_x = car_current_state.kinematics_estimated.position.x_val
        self.car_next_pos_x = car_next_state.kinematics_estimated.position.x_val
        if self.car_current_pos_x == self.car_next_pos_x:
            self.sensing_info.moving_angle = self.airsim_env.get_moving_angle(backed_car_state, car_next_state,
                                                                              way_points,
                                                                              check_point_index)
        else:
            self.sensing_info.moving_angle = self.airsim_env.get_moving_angle(car_current_state, car_next_state,
                                                                              self.way_points,
                                                                              check_point_index)
        collision_info = self.client.simGetCollisionInfo(self.player_name)
        if collision_info.has_collided:
            if self.collision_time_stamp < collision_info.time_stamp:
                self.sensing_info.collided = True
            else:
                self.sensing_info.collided = False
        else:
            self.sensing_info.collided = False
        self.collision_time_stamp = collision_info.time_stamp

        self.sensing_info.lap_progress = self.airsim_env.get_progress(car_next_state, self.way_points,
                                                                      check_point_index, 1, 1)
        self.sensing_info.track_forward_angles = self.airsim_env.get_track_forward_angle(car_next_state,
                                                                                         self.way_points,
                                                                                         check_point_index)
        self.sensing_info.track_forward_obstacles = self.airsim_env.get_track_forward_obstacle(car_next_state,
                                                                                               self.way_points,
                                                                                               check_point_index,
                                                                                               self.all_obstacles)
        return self.sensing_info

    def run(self, time_limit_hour):

        os.makedirs("./save_model/" + str(self.run_cid))
        os.makedirs("./save_graph/" + str(self.run_cid))

        car_prev_state = self.client.getCarState(self.player_name)
        # 조금 주행을 시킨다.
        self.make_initial_movement(self.car_controls, self.client)

        check_point_index = 0
        car_current_state = self.client.getCarState(self.player_name)
        backed_car_state = car_current_state

        scores, episodes = [], []
        current_episode = 0
        scores_per_episode = []
        frozen = 0
        # print("agent_current_state:{}".format(car_current_state))

        finish = False
        time_limit_sec = time_limit_hour * 60 * 60
        # while 루프시작.
        while not finish:
            # 현재 상태 구성
            agent_current_state = self.airsim_env.get_current_state(car_current_state, car_prev_state, self.way_points,
                                                                    check_point_index, self.all_obstacles)
            # print(agent_current_state)
            agent_current_state = np.reshape(agent_current_state, [1, self.state_size])
            check_point_index, _ = self.airsim_env.get_current_way_points(car_current_state, self.way_points,
                                                                          check_point_index)

            # 시뮬레이터에 제어를 넣는다(# 선택한 행동으로 환경에서 한 타임스텝 진행)
            action = self.agent.get_action(agent_current_state)
            self.car_controls = self.interpret_action(action, self.car_controls)
            self.client.setCarControls(self.car_controls)
            time.sleep(self.control_interval)

            # 다음 상태
            car_next_state = self.client.getCarState(self.player_name)
            check_point_index, _ = self.airsim_env.get_current_way_points(car_next_state, self.way_points,
                                                                          check_point_index)
            agent_next_state = self.airsim_env.get_current_state(car_next_state, car_current_state, self.way_points,
                                                                 check_point_index, self.all_obstacles)
            agent_next_state = np.reshape(agent_next_state, [1, self.state_size])

            # 센싱 데이터 계산
            sensing_info = self.calc_sensing_data(car_next_state, car_current_state, backed_car_state, self.way_points,
                                                  check_point_index)

            # 보상 함수로 파라미터를 넘겨준다.
            reward = self.compute_reward(sensing_info)

            # 여기서  done 은 보통은 도로 심하게 이탈해서 더이상 진행하기 어려운 경우.
            # frozen 시뮬레이터가 응답 없는 경우. 리셋.
            done, frozen = self.is_done(car_next_state, car_current_state, reward, frozen)

            scores_per_episode.append(reward)

            if is_debug:
                print("### cur_state", agent_current_state, ",action:", action, ",reward:", reward, ",next_stat:",
                      agent_next_state, done)

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            self.agent.append_sample(agent_current_state, action, reward, agent_next_state, done)

            # 매 타임스텝마다 학습
            if len(self.agent.memory) >= self.agent.train_start:
                self.agent.train_model()

            if done:

                # 한 에피소드가 끝남.
                score = np.sum(scores_per_episode)
                episodes.append(current_episode)

                scores.append(round(score, 2))

                # 추이를 보기 위해서
                graph_x_width = 500
                post_fix = math.floor((len(episodes) - 1) / graph_x_width)
                graph_start = post_fix * graph_x_width
                pylab.plot(episodes[graph_start:], scores[graph_start:], 'b')
                pylab.savefig("./save_graph/" + str(self.run_cid) + "/dqn_graph_" + str(post_fix) + ".png")
                if len(episodes) % graph_x_width == 0:
                    pylab.clf()

                print("Num of steps done :", current_episode, "episode:", current_episode, "  score:", score,
                      "  memory length:",
                      len(self.agent.memory), "  epsilon:", self.agent.epsilon, " check point reached:",
                      check_point_index)

                if current_episode % 10 == 0:
                    self.agent.model.save_weights(
                        "./save_model/" + str(self.run_cid) + "/dqn_weight_" + str(current_episode) + ".h5")

                # 모델 업데이트
                self.agent.update_target_model()

                self.client.reset()
                # 리셋후 조금 주행을 시킨다.
                self.make_initial_movement(self.car_controls, self.client)
                # 변수들 초기화
                check_point_index = 0
                scores_per_episode = []

                current_episode += 1

            if round(self.car_current_pos_x, 4) != round(self.car_next_pos_x, 4):
                backed_car_state = car_current_state
            car_prev_state = car_current_state
            car_current_state = car_next_state

            if time_limit_sec != 0 and time.time() - self.start_time > time_limit_sec:
                finish = True
            ##END OF LOOP

    def is_done(self, car_state, prev_car_state, reward, frozen=0):
        done = 0
        if reward <= -1:
            done = 1
        elif car_state.speed <= 1:
            done = 1
        elif reward == 10:
            done = 1
        elif car_state.kinematics_estimated.position.x_val == prev_car_state.kinematics_estimated.position.x_val and car_state.kinematics_estimated.position.y_val == prev_car_state.kinematics_estimated.position.y_val:
            frozen = frozen + 1
            if frozen > 10:
                frozen = 0
                done = 1
                print("Simulator frozen for some reason ==> Call, done!(reset)")
        return done, frozen

    def interpret_action(self, action, car_controls):
        selected_action = self.action_space()[action]
        car_controls.steering = selected_action['steering']
        car_controls.throttle = selected_action['throttle']
        # if(self.client.getCarState().speed < 20):
        #     car_controls.throttle = 1
        # print(car_controls.throttle)
        return car_controls

    def override_model(self):
        self.agent.model = self.build_custom_model()
        self.agent.target_model = self.build_custom_model()
        # 타깃 모델 초기화
        self.agent.update_target_model()

    @abstractmethod
    def action_space(self):
        raise NotImplementedError('Implement me in subclass')

    @abstractmethod
    def compute_reward(self, sensing_info):
        raise NotImplementedError('Implement me in subclass')

    @abstractmethod
    def build_custom_model(self):
        raise NotImplementedError('Implement me in subclass')

class IncorrectAction(Exception):
    pass
