import setup_path
import sys
import time
import math
import numpy as np

from dqn_custom_client import DQNCustomClient

# =========================================================== #
# Global Configurations
# =========================================================== #
model_weight_path = "./save_model/.../dqn_weight_00.h5"
TRY_LIMIT = 3
current_clock_speed = 1


class DQNEvaluator(DQNCustomClient):

    def __init__(self):
        super().__init__()
        self.start_time = 0
        self.end_time = 0
        # model load
        self.override_model()
        self.agent.load_model(model_weight_path)
        self.control_interval = round(0.1 / current_clock_speed,2)

    # detect simulator frozen
    def is_frozen(self, car_next_state, car_current_state):
        frozen = False
        if (car_next_state.speed == car_current_state.speed) and (
                car_next_state.kinematics_estimated.position.x_val == car_current_state.kinematics_estimated.position.x_val and car_next_state.kinematics_estimated.position.y_val == car_current_state.kinematics_estimated.position.y_val):
            self.frozen_count = self.frozen_count + 1
            if self.frozen_count > 10:
                self.frozen_count = 0
                frozen = True
                print("Simulator frozen for some reason ==> Call, done!(reset)")
        return frozen

    @staticmethod
    def convert_to_minsec(lap_second):
        lap_minute = 0
        if lap_second >= 60:
            lap_minute = math.floor(lap_second / 60)
            lap_second = lap_second % 60
        return lap_minute, round(lap_second, 3)

    def get_lap_sec(self):
        return round(self.end_time - self.start_time, 3)

    def run_model(self, try_limit):

        self.start_time = time.time()

        car_prev_state = self.client.getCarState(self.player_name)
        check_point_index = 0

        try_count = 0
        complete_laps = []
        finish = False
        cur_lab = 1
        half_complete_flag = False

        while not finish:
            done = 0
            # current stat
            car_current_state = self.client.getCarState(self.player_name)
            agent_current_state = self.airsim_env.get_current_state(car_current_state, car_prev_state, self.way_points,
                                                                    check_point_index, self.all_obstacles)
            agent_current_state = np.reshape(agent_current_state, [1, self.state_size])
            check_point_index, _ = self.airsim_env.get_current_way_points(car_current_state, self.way_points,
                                                                          check_point_index)
            # 현재 상태로 행동을 선택한다.
            # 시뮬레이터에 제어를 넣는다(# 선택한 행동으로 환경에서 한 타임스텝 진행)
            action = self.agent.get_eval_action(agent_current_state)
            self.car_controls = self.interpret_action(action, self.car_controls)
            self.client.setCarControls(self.car_controls)
            time.sleep(self.control_interval)

            # 행동 이후의 상태를 구함.
            car_next_state = self.client.getCarState(self.player_name)

            # 여기서  done 은 보통은 도로 심하게 이탈해서 더이상 진행하기 어려운 경우.
            collision_info = self.client.simGetCollisionInfo(self.player_name)
            if collision_info.has_collided:
                if self.collision_time_stamp < collision_info.time_stamp:
                    collided = True
                else:
                    collided = False
            else:
                collided = False
            self.collision_time_stamp = collision_info.time_stamp

            distance_from_center = self.airsim_env.get_distance_from_center(car_next_state, self.way_points,
                                                                            check_point_index)
            # print("distance_from_center:{}".format(distance_from_center))

            # ######################################
            #  Reset Condition
            # - too slow
            # - collision
            # - off the track : = road width/2 + car width/2 (= 1.5)
            # - complete
            # - when simulator frozen : reset.
            #
            frozen = self.is_frozen(car_next_state, car_current_state)

            progress = self.airsim_env.get_progress(car_current_state, self.way_points, check_point_index, cur_lab)
            if progress >= 52:
                half_complete_flag = False
            elif progress >= 50:
                half_complete_flag = True
            if half_complete_flag and progress == 0:
                cur_lab = 2
                progress = 50
            # print("current progress:{}".format(progress))

            if frozen:
                done = 99  # abnormal status
            elif progress > 2 and self.airsim_env.get_speed(car_next_state) <= 1:
                done = 1
            elif collided:
                done = 2
            elif distance_from_center > self.half_road_limit:
                done = 3
            elif progress >= 100:
                done = 10

            if done:
                self.end_time = time.time()

                # count only when normal
                if done != 99:
                    # print("done code : {}".format(done))
                    total_sec = self.get_lap_sec() * current_clock_speed
                    mins, secs = self.convert_to_minsec(total_sec)
                    is_complete = progress >= 100
                    if is_complete:
                        complete_laps.append(total_sec)
                        print(
                            "[{}] the evaluation result -  completed !!, progress :100 % , estimated lap time: {} "
                            "mins. {} secs".format(
                                try_count + 1, mins, secs))
                    else:
                        print(
                            "[{}] the evaluation result -  not completed,  progress {} %".format(
                                try_count + 1, progress))

                    try_count += 1
                    if try_count >= try_limit:
                        finish = True

                check_point_index = 0
                self.client.reset()
                time.sleep(0.2)

                car_current_state = self.client.getCarState(self.player_name)


                self.frozen_count = 0

                self.start_time = time.time()
                self.make_initial_movement(self.car_controls, self.client)

                cur_lab = 1
                half_complete_flag = False
            # saving current status to previous'
            car_prev_state = car_current_state

        # END OF WHILE LOOP
        # ################################

        print("All evaluations are ended.")
        if len(complete_laps) > 0:
            min, sec = self.convert_to_minsec(np.min(complete_laps))
            print("The best lap time is [{} mins. , {} secs.] ".format(min, sec))
        else:
            print("There's no finished lap record yet.")
        self.client.reset()


if __name__ == "__main__":
    eval_client = DQNEvaluator()
    eval_client.run_model(TRY_LIMIT)
    sys.exit()
