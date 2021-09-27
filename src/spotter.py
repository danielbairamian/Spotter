import copy
import cv2
import numpy as np

'''Helper class that will hold per-agent information'''


def stack_renders(agents, env_name):

    stacked_render = np.asarray(agents[0].frame_hist)
    for agent in agents:
        stacked_render = np.minimum(stacked_render, agent.frame_hist)
    video_writer(stacked_render, len(agents), env_name + "_stacked")


def recolor_hist(agent, target_color, new_color):
    for img in agent.frame_hist:
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                pixel = img[row][col]
                if (pixel == target_color).all():
                    img[row][col] = new_color


def video_writer(frame_history, id, env_name):
    height = np.shape(frame_history)[1]
    width = np.shape(frame_history)[2]

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    fps = 30
    video_filename = env_name + '_output'+str(id)+'.avi'
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    for img in frame_history:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out.write(img)
    out.release()


def disable_view_window():
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=True)
        self.window.set_location(0, 0)

    rendering.Viewer.__init__ = constructor


class Spotter_Agent:
    def __init__(self, agent, function_handle, memory_input):
        """disable view window moves renders outside canvas so we don't see them
            There might be a better way to do this"""
        self.agent = agent
        self.function_handle = function_handle
        self.memory_input = memory_input
        self.ep_reward_hist = []
        self.ep_reward = 0
        self.is_done_ep = False
        self.env = None
        self.obs = None
        self.init_state = None
        self.last_frame = None
        self.frame_hist = []

    """sample action according to given function handle"""
    def take_action(self, obs):
        action = getattr(self.agent, self.function_handle)(obs)
        return action

    """repeat last frame of rendering for synchronization"""
    def repeat_frame(self):
        self.frame_hist.append(self.last_frame)


'''Main Spotter class that will sync all runs'''


class Spotter:
    """initialize spotter"""
    """
    Recolor Info: a tuple containing
        - Agent ID - Integer - (position in array starting at 0)
        - Target Pixel Color: The Pixel Color we want to recolor 
        - New Pixel Color: The Pixel Color we want the recolor to be
    """
    def __init__(self, env, env_name, agent_list, function_handle_list, memory_list, individual_render=False,
                 recolor_render=False, recolor_info=None):
        self.env = env
        self.env_name = env_name
        self.ep_max_length = env._max_episode_steps
        self.spotter_agents = []
        self.agent_list = agent_list
        self.function_handle_list = function_handle_list
        self.memory_list = memory_list
        self.individual_render = individual_render
        self.recolor_render = recolor_render
        self.recolor_info = recolor_info
        self.create_agents()
        disable_view_window()

    """create Spotter agents"""
    def create_agents(self):
        for i in range(len(self.agent_list)):
            spotter_agent = Spotter_Agent(self.agent_list[i], self.function_handle_list[i], self.memory_list[i])
            self.spotter_agents.append(spotter_agent)

    """reset agents for every episode"""
    def reset_agents(self):
        init_obs = self.env.reset()
        for agent in self.spotter_agents:
            agent.ep_reward = 0
            agent.is_done_ep = False
            agent.init_state = init_obs
            agent.env = copy.deepcopy(self.env)
            agent.obs = init_obs
            agent.last_frame = None

    """run for number of episode"""
    def run(self, num_eps=5):
        print("Spotter started with", len(self.agent_list), "bros")
        for _ in range(num_eps):
            self.reset_agents()
            self.run_episode()
            self.save_run_hist()

        if self.recolor_render:
            for info in self.recolor_info:
                recolor_hist(self.spotter_agents[info[0]], np.array(info[1]), np.array(info[2]))

        if self.individual_render:
            for i, agent in enumerate(self.spotter_agents):
               video_writer(agent.frame_hist, i, self.env_name)

        stack_renders(self.spotter_agents, self.env_name)

    """per episode run"""
    def run_episode(self):
        for i in range(self.ep_max_length):
            done_counter = 0
            for k, agent in enumerate(self.spotter_agents):
                if not agent.is_done_ep:
                    run_agent_ep(agent)
                else:
                    agent.repeat_frame()
                    done_counter += 1

            """if all agents finished their run, break"""
            if done_counter == len(self.spotter_agents):
                break

    """save the episode reward in the agent's buffer"""
    def save_run_hist(self):
        for agent in self.spotter_agents:
            agent.ep_reward_hist.append(copy.deepcopy(agent.ep_reward))


"""Helper function to run one step of the simulation"""


def run_agent_ep(agent):
    agent.last_frame = agent.env.render(mode="rgb_array")
    agent.frame_hist.append(agent.last_frame)
    if agent.memory_input:
        action = agent.take_action(np.hstack((agent.obs, agent.init_state)))
    else:
        action = agent.take_action(agent.obs)
    obs2, reward, done, _ = agent.env.step(action)
    agent.ep_reward += reward
    agent.obs = obs2
    agent.is_done_ep = done
    if done:
        agent.env.close()
