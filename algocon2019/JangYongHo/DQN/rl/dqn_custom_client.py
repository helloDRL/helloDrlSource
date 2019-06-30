import setup_path
from dqn_model import DQNClient
from dqn_model import DQNParam
import math
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import sys

# =========================================================== #
# Training finish conditions (hour)
# assign training duration by hour : 0(limit less), 1 (an hour), 1.5 (an hour and half) ...
# =========================================================== #
from numpy.f2py.auxfuncs import throw_error

training_duration = 0

# =========================================================== #
# model/weight load option
# =========================================================== #
model_load = False
model_weight_path = "./save_model/.../dqn_weight_00.h5"


# ===========================================================

class DQNCustomClient(DQNClient):
    def __init__(self):
        self.dqn_param = self.make_dqn_param()
        super().__init__(self.dqn_param)

    # =========================================================== #
    # Tuning area (Hyper-parameters for model training)
    # =========================================================== #
    @staticmethod
    def make_dqn_param():
        dqn_param = DQNParam()
        # TODO: 1. parameter tuning
        dqn_param.discount_factor = 0.99 # default: 0.99
        dqn_param.learning_rate = 0.00025 # default: 0.00025
        dqn_param.epsilon = 1.0 # default: 1.0
        dqn_param.epsilon_decay = 0.999 # default: 0.999
        dqn_param.epsilon_min = 0.01 # default: 0.01
        dqn_param.batch_size = 100 # default: 100
        dqn_param.train_start = 1000 # default: 1000
        dqn_param.memory_size = 20000 # default: 20000
        return dqn_param

    # =========================================================== #
    # Action Space (Control Panel)
    # =========================================================== #
    def action_space(self):
        # =========================================================== #
        # Area for writing code
        # =========================================================== #
        # Editing area starts from here
        #
        # TODO: 2. define proper action space
        actions = [
            dict(throttle=1, steering=0), # 파워가속
            dict(throttle=1, steering=0.1),
            dict(throttle=1, steering=0.2),
            dict(throttle=1, steering=-0.1),
            dict(throttle=1, steering=-0.2),
            dict(throttle=0.7, steering=0), # 40정도로 정속주행.. 인데 이미 가속이 많이 되어있으면 대충 70정도 유지하는듯
            dict(throttle=0.72, steering=0.3),
            dict(throttle=0.77, steering=0.6),
            dict(throttle=0.84, steering=1),
            dict(throttle=0.72, steering=-0.3),
            dict(throttle=0.77, steering=-0.6),
            dict(throttle=0.84, steering=-1),
            dict(throttle=0, steering=0), # 파워감속
            dict(throttle=0, steering=0.5),
            dict(throttle=0, steering=-0.5)
        ]
        #
        # Editing area ends
        # ==========================================================#
        return actions

    def compute_reward(self, sensing_info):

        # =========================================================== #
        # Area for writing code
        # =========================================================== #
        # Editing area starts from here
        #
        # TODO: 3. define proper reward system
        # sensing_info.to_middle = 중앙으로부터 거리
        # sensing_info.collided = 충돌
        # sensing_info.speed = 현재속도
        # sensing_info.moving_forward = 정주행하냐 역주행하냐
        # sensing_info.moving_angle = 중앙선 기준으로 어느각도로 주행중인가(대체로 + 가 오른쪽)
        # sensing_info.track_forward_angles = 10m 간격으로 휘어지는 정도. + 면 우측으로 휨
        # sensing_info.lap_progress = 진척률
        # sensing_info.track_forward_obstacles =
        # self.half_road_limit = 도로절반+차 절반 =

        thresh_dist = self.half_road_limit  # 4 wheels off the track
        dist = abs(sensing_info.to_middle)
        if dist > thresh_dist:
            return -1
        elif sensing_info.collided:
            return -1
        elif sensing_info.moving_forward < 0:
            return -1

        reward = sensing_info.speed / 200
        abscurv1 = abs(sensing_info.track_forward_angles[1] - sensing_info.track_forward_angles[0])
        abscurv1 += abs(sensing_info.track_forward_angles[2] - sensing_info.track_forward_angles[1])
        abscurv1 += abs(sensing_info.track_forward_angles[3] - sensing_info.track_forward_angles[2])
        abscurv1 += abs(sensing_info.track_forward_angles[4] - sensing_info.track_forward_angles[3])
        abscurv1 += abs(sensing_info.track_forward_angles[5] - sensing_info.track_forward_angles[4])/2
        abscurv1 = abscurv1 / 90
        abscurv2 = abscurv1
        abscurv2 += abs(sensing_info.track_forward_angles[5] - sensing_info.track_forward_angles[4])/2
        abscurv2 += abs(sensing_info.track_forward_angles[6] - sensing_info.track_forward_angles[5])
        abscurv2 += abs(sensing_info.track_forward_angles[7] - sensing_info.track_forward_angles[6])
        abscurv2 += abs(sensing_info.track_forward_angles[8] - sensing_info.track_forward_angles[7])
        abscurv2 += abs(sensing_info.track_forward_angles[9] - sensing_info.track_forward_angles[8])
        abscurv2 = abscurv2 / 90
        curv1 = sum(sensing_info.track_forward_angles[0:5])
        curv2 = sum(sensing_info.track_forward_angles[0:10])

        sDist = abs(sensing_info.to_middle) - abscurv2 * thresh_dist
        if sDist < 0:
            sDist = sDist/ 5
        print(round(reward,2), end=' : ')
        rewardD = (thresh_dist / (sDist + thresh_dist) - 0.4)
        if rewardD < 0:
            rewardD = rewardD / 5
        else:
            rewardD = rewardD**2
        print(round(rewardD,2), end = ' : ')

        sAngle = abs(sensing_info.moving_angle) - abscurv2 * 45
        if sAngle < 0:
            sAngle = sAngle / 10
        rewardA = (45 / (sAngle + 45) - 0.4)
        if rewardA < 0:
            rewardA = rewardA / 10
        else:
            rewardA = rewardA**2
        print(round(rewardA, 2), end=' = ')
        reward = reward + rewardA + rewardD
        print(round(reward,2))
        #
        # Editing area ends
        # ==========================================================#
        return round(reward,2)

    def build_custom_model(self):
        #TODO: 4. define proper model
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(128, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()

        model.compile(loss='mse', optimizer=Adam(lr=self.dqn_param.learning_rate))

        return model


if __name__ == "__main__":
    client = DQNCustomClient()

    if model_load:
        client.agent.load_model(model_weight_path)

    client.override_model()

    client.run(training_duration)
    sys.exit()
