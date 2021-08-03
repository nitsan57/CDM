import sys
from contextlib import closing

import numpy as np
from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete
import gym
from enum import IntEnum, Enum
from social_rl.gym_multigrid import register


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}


class RGBCOLORS_MINUS_50:
    BLACK = [0, 0, 0]  # -50
    GRAY = [78, 78, 78]  # -50
    RED = [205, 0, 0]  # -50
    GREEN = [0, 205, 0]  # -50
    YELLOW = [205, 205, 0]  # -50
    BLUE = [0, 94, 205]  #
    MAGENTA = [205, 0, 205]  # -50
    CYAN = [0, 205, 205]  # -50


def create_empty_map(size):
    domain_map = []
    abs_size_x = size
    abs_size_y = size
    for i in range(abs_size_y):
        map_row = []
        for j in range(abs_size_x):
            map_row.append('F')
        domain_map.append("".join(map_row))
    return domain_map


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == "G":
                        return True
                    if res[r_new][c_new] != "H":
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(["F", "H"], (size, size), p=[p, 1 - p])
        res[0][0] = "S"
        res[-1][-1] = "G"
        valid = is_valid(res)
    return ["".join(x) for x in res]


class FrozenLakeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the
    park when you made a wild throw that left the frisbee out in the middle of
    the lake. The water is mostly frozen, but there are a few holes where the
    ice has melted. If you step into one of those holes, you'll fall into the
    freezing water. At this time, there's an international frisbee shortage, so
    it's absolutely imperative that you navigate across the lake and retrieve
    the disc. However, the ice is slippery, so you won't always move in the
    direction you intend.
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {"render.modes": ["human", "ansi"]}

    class Actions(IntEnum):
        # Turn left, turn right, move forward
        WEST = 0
        SOUTH = 1
        EAST = 2
        NORTH = 3

    def __init__(self, desc=None, map_name=None, is_slippery=True, size=4, agent_view_size=3, max_steps=300, n_clutter=10, random_z_dim=50, n_agents=1):
        ####PARAMS FOR PAIRED####
        self.agent_view_size = agent_view_size
        self.minigrid_mode = True
        self.size = size
        self.choose_goal_last = False
        self.max_steps = max_steps
        n_clutter = int(size*size*0.6)
        self.n_clutter = n_clutter
        self.random_z_dim = random_z_dim
        self.num_rows = self.size
        self.num_columns = self.size
        self.adversary_action_dim = self.num_rows * self.num_columns
        self.adversary_max_steps = self.n_clutter + 2
        self.fully_observed = False
        self.n_agents = 1

        # INIT MAP PARAMS
        self.is_slippery = is_slippery
        self.s = -1
        dummy_map = create_empty_map(self.size)
        self.total_row_size, self.total_col_size = len(dummy_map), len(dummy_map[0])  # only for full observability

        self.desc = desc = np.asarray(dummy_map, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        self.nA = 4

        self.nS = nrow * ncol

        isd = np.array([1, 0])
        self.isd = isd

        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        # INIT GYM PARAMS

        self.total_row_size, self.total_col_size = len(dummy_map), len(dummy_map[0])  # only for full observability
        num_actions = len(self.Actions)
        self.action_space = gym.spaces.Discrete(num_actions)
        if self.fully_observed:
            obs_image_shape = (self.total_row_size, self.total_col_size, 3)
        else:
            obs_image_shape = (self.agent_view_size * 2, self.agent_view_size * 2, 3)

        if self.minigrid_mode:
            msg = 'Backwards compatibility with minigrid only possible with 1 agent'
            assert self.n_agents == 1, msg

            # Single agent case
            # Images have three dimensions
            self.image_obs_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=obs_image_shape,
                dtype='uint8')

            observation_space = {'image': self.image_obs_space}
            self.observation_space = gym.spaces.Dict(observation_space)

        else:
            print("OTHER MOD NOT SUPPORTED!!")
            assert n_agents == 1

        self.origin_observation_space = self.observation_space
        self.adversary_action_space = gym.spaces.Discrete(self.adversary_action_dim)
        self.adversary_image_obs_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.total_row_size, self.total_col_size, 3),
            dtype='uint8')

        self.adversary_randomz_obs_space = gym.spaces.Box(low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)

        self.adversary_ts_obs_space = gym.spaces.Box(
            low=0, high=self.adversary_max_steps, shape=(1,), dtype='uint8')
        self.adversary_observation_space = gym.spaces.Dict(
            {'image': self.adversary_image_obs_space,
             'time_step': self.adversary_ts_obs_space,
             'random_z': self.adversary_randomz_obs_space})

        #NORMAL INIT##
        self.max_row = self.num_rows - 1
        self.max_col = self.num_columns - 1

        self.step_order = ["choose_goal", "choose_agent", ]  # else choose_hollow_locations
        self.adversary_min_steps_for_init_map = len(self.step_order)
        self.adversary_step_done = False
        self.reset()

    def encode(self, row, col):
        return row * self.ncol + col

    def decode(self, s):
        col = s % self.num_columns
        row = s // self.num_columns
        return row, col

    def after_adversarial(self):
        ncol = self.ncol
        nrow = self.nrow
        desc = self.desc
        is_slippery = self.is_slippery
        P = self.P
        nA = self.nA
        nS = self.nS
        isd = self.isd

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            done = bytes(newletter) in b"GH"
            reward = float(newletter == b"G")
            return newstate, reward, done
        self.all_legal_states_to_random_sample = []
        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        self.all_legal_states_to_random_sample.append(s)
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (1.0 / 3.0, *update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)
        self.observation_space = self.origin_observation_space

    def get_map_image(self, adversarial=False):
        h, w = len(self._str_map), len(self._str_map[0])
        rgb_map = np.zeros((h, w, 3)).astype(np.uint8)
        if self.s != -1:
            taxi_row, taxi_col = self.decode(self.s)
        else:
            taxi_row, taxi_col = -1, -1  # unintilized values
        for r_index, row in enumerate(self._str_map):
            for c_index, char in enumerate(self._str_map[r_index]):
                cell_color = np.array(RGBCOLORS_MINUS_50.BLACK)  # BLACK-50
                if char == "F" or char == "S":
                    cell_color = np.array(RGBCOLORS_MINUS_50.BLUE)  # gray-50
                elif char == "H":
                    cell_color = np.array(RGBCOLORS_MINUS_50.BLACK)  # gray-50
                elif char == "G":
                    cell_color = np.array(RGBCOLORS_MINUS_50.GREEN)  # gray-50
                if (taxi_row, taxi_col) == (r_index, c_index):
                    cell_color += RGBCOLORS_MINUS_50.YELLOW
                    cell_color = (cell_color / 2).astype(np.uint8)

                rgb_map[r_index, c_index] = cell_color
        if not self.fully_observed and not adversarial and taxi_row != -1:
            row, col = taxi_row, taxi_col
            min_row, max_row = max(row - self.agent_view_size, 0), min(row + self.agent_view_size, self.total_row_size)
            min_col, max_col = max(col - self.agent_view_size, 0), min(col + self.agent_view_size, self.total_col_size)
            rgb_map[min_row:max_row, min_col:max_col] = rgb_map[min_row:max_row, min_col:max_col] + 50
        else:
            rgb_map = rgb_map + 50

        return rgb_map

    def update_map(self, loc_row, loc_column, char):
        temp_row = list(self._str_map[loc_row])  # to list
        temp_row[loc_column] = char
        self._str_map[loc_row] = "".join(temp_row)
        self.desc = np.asarray(self._str_map, dtype='c')

    def reset(self):
        """Fully resets the environment to an empty grid with no agent or goal."""
        self.n_clutter_placed = 0
        self.adversary_step_count = 0
        self._str_map = create_empty_map(self.size)
        self.desc = np.asarray(self._str_map, dtype='c')
        self.lastaction = None

        # Extra metrics
        self.reset_metrics()

        image = self.get_map_image(adversarial=True)
        self.step_count = 0
        obs = {
            'image': image,
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z()
        }

        return obs

    def reset_agent(self):
        """Resets the agent's start position, but leaves goal and walls."""
        # Remove the previous agents from the world
        if self.adversary_step_count >= self.adversary_min_steps_for_init_map:
            self.adversary_step_done = True
            self.compute_shortest_path()
            self.after_adversarial()
        else:
            print("Error trying to reset agent before making adversarial step")

        self.step_count = 0
        self.s = self.init_s
        self.lastaction = None

        # Return first observation
        return self.get_observation()

    def step(self, actions):

        if type(actions) != int:
            a = actions[0]
        else:
            a = actions
        assert type(actions) == int or len(actions) == 1

        transitions = self.P[self.s][a]
        i = discrete.categorical_sample([t[0] for t in transitions], self.np_random)  # only 0 index is chosen
        p, s, r, done = transitions[i]
        self.s = s
        self.lastaction = a

        obs = self.get_observation()

        self.step_count += 1

        if self.step_count >= self.max_steps:
            done = True

        rewards = np.array([r])

        if self.minigrid_mode:
            rewards = rewards[0]
        return obs, rewards, done, {"prob": p}

    # def find_free_space_and_put(self, symbol):
    #     free_indices = [i for i in range(self.num_columns * self.num_rows) if self.get_map_symbol(self.unpack_index(i)[0], self.unpack_index(i)[1], False) == " "]
    #     index = np.random.choice(free_indices)
    #     loc_row, loc_col = self.unpack_index(index)
    #     self.put_in_map(loc_row, loc_col, symbol)

    def step_adversary(self, loc):
        """The adversary gets n_clutter + 2 moves to place the goal, agent, blocks.

        The action space is the number of possible squares in the grid. The squares
        are numbered from left to right, top to bottom.

        Args:
          loc: An integer specifying the location to place the next object which
            must be decoded into x, y playable_coordinates.

        Returns:
          Standard RL observation, reward (always 0), done, and info
        """
        step_order = self.step_order  # ["choose_goal", "choose_agnet"]  # else hollow places
        current_turn = step_order[self.adversary_step_count] if self.adversary_step_count < len(step_order) else "place_hollow"
        if loc >= self.adversary_action_dim:
            raise ValueError('Position passed to step_adversary is outside the grid.')

        # Add offset of 1 for outside walls
        row, col = self.decode(loc)

        done = False

        # Place goal
        if current_turn == "choose_goal":
            self.update_map(row, col, "G")

        # Place the agent
        elif current_turn == "choose_agent":
            # self.init_taxi_row, self.init_taxi_col = [row, col]
            self.deliberate_agent_placement = 1
            self.update_map(row, col, "S")
            self.init_s = self.encode(row, col)

        # Place hollow
        elif self.adversary_step_count < self.adversary_max_steps:
            # If there is already an object there, action does nothing, also if it is on the grid bounderies
            agent_row, agent_col = self.decode(self.init_s)
            if (row, col) != (agent_row, agent_col):
                self.update_map(row, col, "H")
            else:
                indices = [i for i in range(self.size * self.size) if i != row * col]
                chosen_index = np.random.choice(indices)
                new_r, new_c = self.decode(chosen_index)
                self.update_map(new_r, new_c, "H")
            self.n_clutter_placed += 1

        self.adversary_step_count += 1

        # End of episode
        if self.adversary_step_count >= self.adversary_max_steps:
            done = True
            self.reset_agent()

        image = self.get_map_image(adversarial=True)
        obs = {
            'image': image,
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z()
        }

        return obs, 0, done, {}

    def compute_shortest_path(self):
        "Currently supports single agent only"
        # if len(self.agent_starting_location) == 0 or len(self.passangers_start_locations) == 0 or len(self.passengers_destinations) == 0:
        # return

        self.distance_to_goal = 0  # abs(
        # self.passengers_destinations[0][0] - self.passangers_start_locations[0][0]) + abs(
        #   self.passengers_destinations[0][1] - self.passangers_start_locations[0][1]) + abs(self.passangers_start_locations[0][0] - self.agent_starting_location[0][0]) + abs(self.passangers_start_locations[0][1] - self.agent_starting_location[0][1])

        self.passable = -1
        # NOT ENOUGH

    def get_observation(self) -> np.array:
        """
        Takes only the observation of the specified agent.
        Args:
            state: state of the domain (taxis, fuels, passengers_start_playable_coordinates, destinations, passangers_locations)
            agent_name: observer name
            agent_view_size: the size that the agent can see in the map (around it) in terms of other txis

        Returns: observation of the specified agent (state wise)

        """

        image = self.get_map_image()
        agent_image = np.copy(image)
        row, col = self.decode(self.s)

        if not self.fully_observed:
            agent_image = np.zeros((self.agent_view_size * 2, self.agent_view_size * 2, 3)).astype(np.uint8)
            gray = np.array([78] * 3) + 50  # gray - show walls in edges
            agent_image = agent_image + gray
            min_row, max_row = max(row - self.agent_view_size, 0), min(row + self.agent_view_size, self.total_row_size - 1)
            min_col, max_col = max(col - self.agent_view_size, 0), min(col + self.agent_view_size, self.total_col_size - 1)
            sliced_image = image[min_row:max_row, min_col:max_col]
            start_row = 0
            end_row = self.agent_view_size * 2
            start_col = 0
            end_col = self.agent_view_size * 2
            if sliced_image.shape[0] != self.agent_view_size * 2:
                if max_row == self.total_row_size - 1:
                    start_row = 0
                    end_row = sliced_image.shape[0]
                if min_row == 0:
                    start_row = self.agent_view_size * 2 - sliced_image.shape[0]
                    end_row = self.agent_view_size * 2
            if sliced_image.shape[1] != self.agent_view_size * 2:
                if max_col == self.total_col_size - 1:
                    start_col = 0
                    end_col = sliced_image.shape[1]
                if min_col == 0:
                    start_col = self.agent_view_size * 2 - sliced_image.shape[1]
                    end_col = self.agent_view_size * 2
            agent_image[start_row: end_row, start_col:end_col] = sliced_image  # [start_row: end_row, start_col:end_col]

        if not self.minigrid_mode:
            agent_image = [agent_image]

        obs = {
            'image': agent_image}
        return obs

    def get_goal_x(self):
        return 0

    def get_goal_y(self):
        return 0

    def reset_metrics(self):
        self.distance_to_goal = -1
        self.n_clutter_placed = 0
        self.deliberate_agent_placement = -1
        self.passable = -1
        self.shortest_path_length = (self.num_columns) * (self.num_rows) + 1

    def sample_random_state(self, seed=None):
        # np.random.choice
        temp_s = self.s
        r_s = self.np_random.choice(self.all_legal_states_to_random_sample)
        self.s = r_s
        obs = self.get_observation()
        self.s = temp_s
        return obs

    def render(self, mode="human"):
        if mode == "rgb_array":
            return self.get_map_image()
        outfile = StringIO() if mode == "ansi" else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write(
                "  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction])
            )
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()

    def generate_random_z(self):
        return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)

    def init_print_mode(self):
        all_map = []
        line_list = []

        for row, line in enumerate(self._str_map):
            for col, char in enumerate(line):
                line_list.append(char)
            line = "".join(line_list)
            all_map.append(line + '\n')
            line_list = []
        return "".join(all_map)

    def __str__(self):
        if self.adversary_step_done:
            return self.render("ansi")
        else:
            return self.init_print_mode()


if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname

register.register(
    env_id='MultiGrid-FrozenEnv-v0',
    entry_point=module_path + ':FrozenLakeEnv'
)


# if __name__ == '__main__':
#     env = FrozenLakeEnv()
#     # debug_print(env._str_map)
#     for i in range(3):
#         env.step_adversary(2 * i)
#     env.reset_agent()
#     env.render()
#     # debug_print(env._str_map)

#     new_env = env
#     new_env.reset_agent()
#     new_env.render()
#     for _ in range(50):
#         next_s, r, done, prob = new_env.step(1)
#     new_env.render()
#     print(new_env.s, r)
