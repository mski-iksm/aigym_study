import numpy as np
import dqn_libs
from time import sleep

"""
https://qiita.com/sugulu/items/3c7d6cbe600d455e853b
http://neuro-educator.com/rl2/
https://elix-tech.github.io/ja/2016/06/29/dqn-ja.html
"""

# set model parameters =========================================
memory_d = 10000  # record size of replay memories
episode_nums = 1000000  # number of episodes
enough_thresh = int(memory_d * 0.5)  # record numbers to determine enough records
batch_size = 32

epsilon_0 = 0.99  # initial epsilon
gamma = 0.99  # time discount
alpha = 0.0001  # learning rate

goal_steps = 195
visual_goal_steps = 100

learned = 0

# list for calculating max steps in each episode
monitor_step_nums = 10
tripped_step = np.zeros(monitor_step_nums)


# model main part ==============================================
# initiate Q and target network parameters
dqnbase = dqn_libs.DqnBase(epsilon_0=epsilon_0, gamma=gamma, alpha=alpha,
                           memory_d=memory_d, batch_size=batch_size)

save_index = 0

# episode loop
for ep in range(episode_nums):
    print("episode:{} ==================================".format(ep))
    # # set initial state
    state = dqnbase.reset_env()
    reward = 1.0
    done = False
    info = {}

    # # while not done
    t = 0
    while not done:
        t += 1
        # ## get action
        next_action = dqnbase.get_action(state=state, episode=ep)

        # ## get next step state after action
        next_state, reward, done, info = dqnbase.run_step(next_action=next_action)
        if done:
            next_state = np.zeros(next_state.shape)  # 次の状態s_{t+1}はない
            if t < goal_steps:
                reward = -200  # penalty when tripped
            else:
                reward = 1  # 立ったまま195step超えて終了時は報酬

            # save number of steps
            print("ep {} end with {} steps".format(ep, t))
            tripped_step[ep % monitor_step_nums] = t
        else:
            reward = 0  # 各ステップで立ってたら報酬追加

        # ## append action and next state to D
        dqnbase.save_states(save_index % memory_d, state, next_action, reward, next_state, done)

        # ## sample minibatch if D has enough records
        if save_index > enough_thresh:

            learned = np.mean(tripped_step) > visual_goal_steps
            if learned and ep % 30 == 0:
                dqnbase.env.render()
                sleep(0.01)

            if save_index >= memory_d:
                max_memory_index = memory_d
            else:
                max_memory_index = save_index
            # sample_st, sample_at, sample_rt, sample_st1, sample_done
            sample_data = dqnbase.sample_records(max_memory_index)

            # get target value
            target_value = dqnbase.get_target_value(sample_data)

            # ### correct parameters of Q network
            dqnbase.train_network(state=sample_data[0], action=sample_data[1],
                                  target_value=target_value)

        state = next_state
        save_index = save_index + 1
    # ### copy Q network parameters to target value (sometimes)
    dqnbase.copy_q_to_target()
