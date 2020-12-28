import gym
from baselines.Actor_Critic.PPO import PPO
from baselines.Imitation_Learning.GAIL import GAIL
import torch
from Spotter.src.spotter import Spotter

if __name__ == '__main__':
    """
    Inputs:

    OpenAI gym environment

    List of agents

    Function handle of action sampler
        --> Function call of agent:
            --> Input: OpenAI Observation
            --> Output: OpenAI Action Value
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    '''gym environment'''
    env_name = "CartPole"
    env = gym.make(env_name + "-v1")

    """agent list"""

    '''PPO Agent'''
    agent_1 = PPO(env          = env,
                  env_name     = env_name,
                  device       = device,
                  agent_actor  = None,
                  agent_critic = None,
                  agent_buffer = None)


    agent_1.load_model()

    """GAIL Agent trained to imitate the PPO Agent"""
    agent_2 = GAIL(env=env,
                   env_name      =env_name,
                   device        =device,
                   expert        = None,
                   student_actor = None,
                   student_critic= None,
                   discriminator = None,
                   expert_buffer = None,
                   student_buffer= None)

    agent_2.load_model()

    agent_list = [agent_1, agent_2]

    """function handle of agents"""
    function_handle_list = ["get_greedy_action", "get_student_greedy_action"]

    """initialize spotter and run"""
    spotter = Spotter(env, agent_list, function_handle_list)
    spotter.run()
