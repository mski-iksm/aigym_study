# coding:utf-8
"""
playing CartPole with Q learning.

coding based on http://neuro-educator.com/rl1/
"""

import gym  # 倒立振子(cartpole)の実行環境
from gym import wrappers  # gymの画像保存
import numpy as np
import time

# set parameter
num_dizitized = 6
epsilon_0 = 0.5

# epsilon parameter
gamma = 0.99
alpha = 0.5


# set parameter for main training
env = gym.make('CartPole-v0')
max_number_of_steps = 200  # number of steps in an episode
num_consecutive_iterations = 100
num_episodes = 2000  # total try count
goal_average_reward = 195  # goal reward (succcess threshold)
q_table = np.random.uniform(
    low=-1, high=1, size=(num_dizitized**4, env.action_space.n))
# row: digitized parms, col: action

total_reward_vec = np.zeros(num_consecutive_iterations)  # store reward
final_x = np.zeros((num_episodes, 1))  # 学習後、各試行のt=200でのｘの位置を格納
islearned = 0  # learn end flag
isrender = 0  # draw flag


def bins(clip_min, clip_max, num):
    """
    discretize observation values
    """
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]


def digitize_state(observation):
    """
    digitize values
    observation[0]: cart position [-2.4, 2.4]
    observation[1]: cart velocity [-3.0, 3.0]
    observation[2]: pole angle    [-0.5, 0.5]
    observation[3]: pole velocity [-2.0, 2.0]

    digitize each value and store as num_dizitized base decimal
    [-2.4,-3,-0.5,-2.] --> 0
    [2.4,3,0.5,2.] --> 5 + 5*6 + 5*6^2 + 5*6^3 = 1295
    """
    cart_pos, cart_v, pole_angle, pole_v = observation
    digitized = [
        np.digitize(cart_pos, bins=bins(-2.4, 2.4, num_dizitized)),
        np.digitize(cart_v, bins=bins(-3.0, 3.0, num_dizitized)),
        np.digitize(pole_angle, bins=bins(-0.5, 0.5, num_dizitized)),
        np.digitize(pole_v, bins=bins(-2.0, 2.0, num_dizitized))
    ]
    return sum([x * (num_dizitized**i) for i, x in enumerate(digitized)])


def get_action(next_state, episode):
    """
    Function to calculate next action
    ε-greedy algorithem
    """

    epsilon = epsilon_0 * (1 / (episode + 1))
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])
    else:
        next_action = np.random.choice([0, 1])
    return next_action


def update_qtable(q_table, state, action, reward, next_state):
    """
    Function to update q_table
    """

    next_max_q = max(q_table[next_state][0], q_table[next_state][1])  # max q of 2 actions
    q_table[state, action] = (1 - alpha) * q_table[state, action] +\
        alpha * (reward + gamma * next_max_q)

    return q_table


for episode in range(num_episodes):
    # initiate environment
    observation = env.reset()
    state = digitize_state(observation)
    action = np.argmax(q_table[state])
    episode_reward = 0

    for t in range(max_number_of_steps):  # single step iteration
        if islearned == 1:  # draw cartPole when learned
            env.render()
            time.sleep(0.01)
            print(observation[0])  # print x loaction of cart

        # simulate conditions after action
        observation, reward, done, info = env.step(action)

        # modify reward
        if done:
            if t < 195:
                reward = -200  # penalty if tripped
            else:
                reward = 1  # no penalty when ended with pole standing
        else:
            reward = 1  # add reward at each step when stand

        episode_reward += reward  # add reward to eqisode total rewards

        # digitize observation
        next_state = digitize_state(observation)
        # update qtable
        q_table = update_qtable(q_table, state, action, reward, next_state)

        # calculate next action
        action = get_action(next_state, episode)

        # move to next state
        state = next_state

        # process at end
        if done:
            print('%d Episode finished after %f time steps / mean %f' %
                  (episode, t + 1, total_reward_vec.mean()))
            total_reward_vec = np.hstack((total_reward_vec[1:],
                                          episode_reward))  # store reward
            if islearned == 1:  # store last x when train is ended
                final_x[episode, 0] = observation[0]
            break

    if (total_reward_vec.mean() >=
            goal_average_reward):  # success
        print('Episode %d train agent successfuly!' % episode)
        islearned = 1
        # np.savetxt('learned_Q_table.csv',q_table, delimiter=",") #Qtableの保存する場合
        if isrender == 0:
            # env = wrappers.Monitor(env, './movie/cartpole-experiment-1') #動画保存する場合
            isrender = 1
    # 10エピソードだけでどんな挙動になるのか見たかったら、以下のコメントを外す
    # if episode>10:
    #    if isrender == 0:
    #        env = wrappers.Monitor(env, './movie/cartpole-experiment-1') #動画保存する場合
    #        isrender = 1
    #    islearned=1;

if islearned:
    np.savetxt('final_x.csv', final_x, delimiter=",")
