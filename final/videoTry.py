import gym
import mujoco_py
import neat
import pickle
from gym.wrappers import Monitor

env = gym.make('Swimmer-v2')

env = Monitor(env, './video', force=True)
env.seed(0)

max_reward = 0
for i_episode in range(100):
    observation = env.reset()
    trial_fit = 0
    done = False
    for t in range(30):
        #env.render(mode="human")
        env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        trial_fit += reward
        #print("done?: "+str(done)+", reward: "+str(reward))
        if done:
            break
            #print("Episode {} finished after {} timesteps with fitness {}".format(i_episode+1, t+1, reward))
            
    if(trial_fit>=max_reward):
        max_reward = trial_fit
    trial_fit = 0
env.close()
print(max_reward)
