import copy

'''Helper class that will hold per-agent information'''

def disable_view_window():
    from gym.envs.classic_control import rendering
    org_constructor = rendering.Viewer.__init__

    def constructor(self, *args, **kwargs):
        org_constructor(self, *args, **kwargs)
        self.window.set_visible(visible=True)
        self.window.set_location(-10000, -10000)

    rendering.Viewer.__init__ = constructor


class Spotter_Agent:
    def __init__(self, agent, function_handle, env, init_obs):
        """disable view window moves renderes outside canvas so we don't see them
            There might be a better way to do this"""
        disable_view_window()
        self.agent = agent
        self.function_handle = function_handle
        self.env = env
        self.obs = init_obs
        self.ep_reward = 0
        self.is_done_ep = False

    """sample action according to given function handle"""
    def take_action(self, obs):
        action = getattr(self.agent, self.function_handle)(obs)
        return action


'''Main Spotter class that will sync all runs'''


class Spotter:
    """initialize spotter"""
    def __init__(self, env, agent_list, function_handle_list):
        self.env = env
        self.ep_max_length = env._max_episode_steps
        self.spotter_agents = []
        self.agent_list = agent_list
        self.function_handle_list = function_handle_list

    """reset agents for every episode"""
    def reset_agents(self):
        init_obs = self.env.reset()
        self.spotter_agents = []
        for i in range(len(self.agent_list)):
            agent_env = copy.deepcopy(self.env)
            spotter_agent = Spotter_Agent(self.agent_list[i], self.function_handle_list[i], agent_env, init_obs)
            self.spotter_agents.append(spotter_agent)

    """run for number of episode"""
    def run(self, num_eps=10):
        print("Spotter started with", len(self.agent_list), "bros")
        for _ in range(num_eps):
            self.reset_agents()
            self.run_episode()

    """per episode run"""
    def run_episode(self):
        for i in range(self.ep_max_length+1):
            done_counter = 0
            for k, agent in enumerate(self.spotter_agents):
                if not agent.is_done_ep:
                    run_agent_ep(agent)
                else:
                    done_counter += 1

            """if all agents finished their run, break"""
            if done_counter == len(self.spotter_agents):
                break


"""Helper function to run one step of the simulation"""

def run_agent_ep(agent):
    output = agent.env.render(mode="rgb_array")
    action = agent.take_action(agent.obs)
    obs2, reward, done, _ = agent.env.step(action)
    agent.ep_reward += reward
    agent.obs = obs2
    agent.is_done_ep = done
    if done:
        agent.env.close()

