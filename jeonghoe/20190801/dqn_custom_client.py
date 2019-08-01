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
training_duration = 0

# =========================================================== #
# model/weight load option
# =========================================================== #
model_load = True
model_weight_path = "./save_model/T0731_234751/dqn_weight_380.h5"


# ===========================================================

class DQNCustomClient(DQNClient):
    def __init__(self):
        self.dqn_param = self.make_dqn_param()
        super().__init__(self.dqn_param)
        self.best_score = 0
        self.pre_angle = 0
        self.max_lab_progress = 0

    # =========================================================== #
    # Tuning area (Hyper-parameters for model training)
    # =========================================================== #
    @staticmethod
    def make_dqn_param():
        dqn_param = DQNParam()
        dqn_param.discount_factor = 0.99
        dqn_param.learning_rate = 0.00025
        dqn_param.epsilon = 0.2
        dqn_param.epsilon_decay = 0.9999
        dqn_param.epsilon_min = 0.01
        dqn_param.batch_size = 100
        dqn_param.train_start = 1000
        dqn_param.memory_size = 20000
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
        actions = [
            dict(throttle=0.7, steering=0.1),
            dict(throttle=0.7, steering=-0.1),
            dict(throttle=0.7, steering=0.2),
            dict(throttle=0.7, steering=-0.2),
            dict(throttle=0.7, steering=0.3),
            dict(throttle=0.7, steering=-0.3),
            dict(throttle=0.7, steering=0.4),
            dict(throttle=0.7, steering=-0.4),
            dict(throttle=0.7, steering=0.5),
            dict(throttle=0.7, steering=-0.5),
            dict(throttle=0.7, steering=0.6),
            dict(throttle=0.7, steering=-0.6),
            dict(throttle=0.7, steering=0.7),
            dict(throttle=0.7, steering=-0.7),
            dict(throttle=0.7, steering=0),
            dict(throttle=0.8, steering=0),
            dict(throttle=0.9, steering=0),
            dict(throttle=1.0, steering=0)
        ]
        #
        # Editing area ends
        # ==========================================================#
        return actions

    # =========================================================== #
    # Reward Function
    # =========================================================== #
    def compute_reward(self, sensing_info):

        # =========================================================== #
        # Area for writing code
        # =========================================================== #
        # Editing area starts from here
        #

        # 1.중앙 따라가게하기
        thresh_dist = self.half_road_limit  # 4 wheels off the track
        dist = abs(sensing_info.to_middle)
        reward = 0
        dist_reward = 0
        if dist > thresh_dist - 0.1:
            reward = -1
            return reward
        elif sensing_info.collided:
            reward = -1
            return reward
        else:
            if dist > 5:
                dist_reward = 0
            else:
                dist_reward = round((5- round(dist,0))/5,2) * 10
        
        # 2.안흔들리게 하기
        angle = abs(sensing_info.moving_angle) 
        angle_reward = 0
        if angle>=50:
            angle_reward = 0
        elif angle>=40:
            angle_reward = 0.2
        elif angle>=30:
            angle_reward = 0.4
        elif angle>=20: 
            angle_reward = 0.6
        elif angle>=10:
            angle_reward = 0.8
        else:
            angle_reward =1.5

        angle_reward *= 5
        '''    
        if angle >= 1:
            angle_reward = 0
        else:
            angle_reward = 5 * round(1 - angle,2)
        '''
        

        # 3.속도 빠르게 하기 
        # get reward if speed is faster than before
        speed = sensing_info.speed
        speed_reward = 0
        if speed <= 40:
            speed_reward = 0
        elif speed <= 50:    
            speed_reward = 0.1
        elif speed <= 60:
            speed_reward = 0.3
        elif speed <= 70:
            speed_reward = 0.5
        elif speed <= 80:
            speed_reward = 0.7
        else:
            speed_reward = 1

            #speed_reward = round(round(speed, 0)/100,2) 

        reward += dist_reward + angle_reward + speed_reward 

        return reward

    # =========================================================== #
    # Model network
    # =========================================================== #
    def build_custom_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()

        model.compile(loss='mse', optimizer=Adam(lr=self.dqn_param.learning_rate))

        return model


if __name__ == "__main__":
    client = DQNCustomClient()

    client.override_model()

    if model_load:
        client.agent.load_model(model_weight_path)

    client.run(training_duration)
    sys.exit()
