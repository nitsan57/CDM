from hashlib import new
import sys
from contextlib import closing
from io import StringIO
from gym import utils
import gym
from gym.envs.toy_text import discrete
import numpy as np
import gym
from gym.utils import seeding
from social_rl.gym_multigrid import register
from enum import IntEnum, Enum
from scipy.spatial.distance import cityblock


class RGBCOLORS_MINUS_50:
    BLACK = [0, 0, 0]  # -50
    GRAY = [78, 78, 78]  # -50
    RED = [205, 0, 0]  # -50
    GREEN = [0, 205, 0]  # -50
    YELLOW = [205, 205, 0]  # -50
    BLUE = [0, 94, 205]  #
    MAGENTA = [205, 0, 205]  # -50
    CYAN = [0, 205, 205]  # -50


MAP = [
    "+---------+",
    "| : | :G: |",
    "| : | : : |",
    "| : : : : |",
    "| : : | : |",
    "|Y: :F| : |",
    "+---------+",
]

SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF, REFUEL = 0, 1, 2, 3, 4, 5, 6
PASSENGER_IN_TAXI = -1
STEP_REWARD, PICKUP_REWARD, BAD_PICKUP_REWARD, DROPOFF_REWARD, BAD_DROPOFF_REWARD, REFUEL_REWARD, BAD_REFUEL_REWARD, NO_FUEL_REWARD = "step", "good_pickup", "bad_pickup", "good_dropoff", "bad_dropoff", "good_refuel", "bad_refuel", "no_fuel"
MAX_FUEL = 50
REWARD_DICT = {STEP_REWARD: -1,
               PICKUP_REWARD: 0, BAD_PICKUP_REWARD: -10,
               DROPOFF_REWARD: 20, BAD_DROPOFF_REWARD: -10,
               REFUEL_REWARD: 10, BAD_REFUEL_REWARD: -10, NO_FUEL_REWARD: -100}

ACTIONS = [SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF, REFUEL]


# class ActionsEnum(IntEnum):
#     # Turn left, turn right, move forward
#     SOUTH = SOUTH
#     NORTH = NORTH
#     EAST = EAST
#     WEST = WEST
#     PICKUP = PICKUP
#     DROPOFF = DROPOFF
#     REFUEL = REFUEL


class SingleTaxiEnv(discrete.DiscreteEnv):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich
    Description:
    There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off at a random square and the passenger is at a random location. The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off, the episode ends.
    Observations:
    There are 500 discrete states since there are 25 taxi positions, 5 possible locations of the passenger (including the case when the passenger is in the taxi), and 4 destination locations.
    Passenger locations:
    - 2: Y(ellow)
    - 4: in taxi
    Destinations:
    - 1: G(reen)
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
    Rewards:
    There is a default per-step reward of -1,
    except for delivering the passenger, which is +20,
    or executing "pickup" and "drop-off" actions illegally, which is -10.
    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (Y): locations for passengers
    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    class Actions(IntEnum):
        # Turn left, turn right, move forward
        SOUTH = SOUTH
        NORTH = NORTH
        EAST = EAST
        WEST = WEST
        PICKUP = PICKUP
        DROPOFF = DROPOFF
        REFUEL = REFUEL

    def __init__(self, size=5, agent_view_size=3, max_steps=300, n_clutter=10, random_z_dim=50, n_agents=1):
        ####PARAMS FOR PAIRED####
        self.agent_view_size = agent_view_size
        self.minigrid_mode = True
        self.size = size
        self.choose_goal_last = False
        self.max_steps = max_steps
        self.n_clutter = n_clutter
        self.random_z_dim = random_z_dim
        self.num_rows = self.size
        self.num_columns = self.size
        self.adversary_action_dim = self.num_rows * self.num_columns
        self.adversary_max_steps = self.n_clutter + 2
        self.fully_observed = False
        self.n_agents = 1
        # INIT MAP PARAMS
        self.s = -1
        dummy_map = self.create_empty_map(self.size)
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

            self.fuel_space = gym.spaces.Discrete(MAX_FUEL - 1)
            # self.fuel_space = gym.spaces.Box(
            #     low=0, high=MAX_FUEL, shape=(self.n_agents,), dtype='uint8')

            observation_space = {'image': self.image_obs_space, 'fuel': self.fuel_space}
            self.observation_space = gym.spaces.Dict(observation_space)

        else:
            print("OTHER MOD NOT SUPPORTED!!")
            assert n_agents == 1

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

        self.step_order = ["choose_goal", "choose_passanger", "choose_fuel", "choose_agent"]  # else choose_walls
        self.adversary_min_steps_for_init_map = len(self.step_order)
        self.adversary_step_done = False

        self.reset()

    def init_after_adverserial(self):
        self.last_action = None
        self.passengers_locations, self.destination_location, self.fuel_station = self.get_info_from_map()
        self.taxi_fuel = MAX_FUEL
        self.num_states = 5 * 5 * 1 * 2 * MAX_FUEL  # rows, cols, ONE DEST LOC, 2 PASSANGER POS        #25*4*5 * MAX_FUEL

        self.initial_state_distrib = np.zeros(self.num_states)
        self.num_actions = len(ACTIONS)
        self.passenger_in_taxi = len(self.passengers_locations)
        self.P = {state: {action: [] for action in range(self.num_actions)} for state in range(self.num_states)}
        self.all_legal_states_to_random_sample = []
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                for pass_idx in range(len(self.passengers_locations) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(self.destination_location)):
                        for fuel in range(MAX_FUEL):
                            init_fuel = fuel
                            state = self.encode(row, col, pass_idx, dest_idx, fuel)
                            if self.is_possible_initial_state(pass_idx, dest_idx, row, col, fuel):
                                self.all_legal_states_to_random_sample.append(state)
                            for action in range(self.num_actions):
                                # defaults
                                new_row, new_col, new_pass_idx = row, col, pass_idx
                                reward = REWARD_DICT[STEP_REWARD]  # default reward when there is no pickup/dropoff
                                done = False
                                taxi_loc = (row, col)

                                if action == SOUTH and fuel != 0:
                                    new_row = min(row + 1, self.max_row)
                                    if new_row == row + 1:
                                        fuel -= 1
                                elif action == NORTH and fuel != 0:
                                    new_row = max(row - 1, 0)
                                    if new_row == row - 1:
                                        fuel -= 1
                                elif action == EAST and self.desc[1 + row, 2 * col + 2] == b":" and fuel != 0:
                                    new_col = min(col + 1, self.max_col)
                                    if new_col == col + 1:
                                        fuel -= 1
                                elif action == WEST and self.desc[1 + row, 2 * col] == b":" and fuel != 0:
                                    new_col = max(col - 1, 0)
                                    if new_col == col - 1:
                                        fuel -= 1
                                elif action == PICKUP:  # pickup
                                    if pass_idx < self.passenger_in_taxi and taxi_loc == self.passengers_locations[
                                            pass_idx]:
                                        new_pass_idx = self.passenger_in_taxi
                                    else:  # passenger not at location
                                        reward = REWARD_DICT[BAD_PICKUP_REWARD]
                                elif action == DROPOFF:  # dropoff
                                    if (taxi_loc == self.destination_location[
                                            dest_idx]) and pass_idx == self.passenger_in_taxi:
                                        new_pass_idx = dest_idx
                                        done = True
                                        reward = REWARD_DICT[DROPOFF_REWARD]
                                    elif (taxi_loc in self.passengers_locations) and pass_idx == self.passenger_in_taxi:
                                        new_pass_idx = self.passengers_locations.index(taxi_loc)
                                    else:  # dropoff at wrong location
                                        reward = REWARD_DICT[BAD_DROPOFF_REWARD]
                                elif action == REFUEL:
                                    if taxi_loc == self.fuel_station:
                                        reward = REWARD_DICT[REFUEL_REWARD]
                                        fuel = MAX_FUEL - 1
                                    else:
                                        reward = REWARD_DICT[BAD_REFUEL_REWARD]
                                elif fuel == 0:
                                    done = True
                                    reward = REWARD_DICT[NO_FUEL_REWARD]

                                new_state = self.encode(new_row, new_col, new_pass_idx, dest_idx, fuel)
                                self.P[state][action].append((1.0, new_state, reward, done))
                                fuel = init_fuel

        self.init_s = self.encode(self.init_taxi_row, self.init_taxi_col, 0, 0, (MAX_FUEL - 1))
        if self.is_possible_initial_state(0, 0, self.init_taxi_row, self.init_taxi_col, MAX_FUEL - 1):
            self.initial_state_distrib[self.init_s] = 1

        else:
            pass
            # print(self.init_print_mode())
            print("BUG IN ADVERSERAIL INIT!!! NOT LEGAL STATE", 0, 0, self.init_taxi_row, self.init_taxi_col, MAX_FUEL - 1, self.destination_location, self.passengers_locations)

        self.s = self.init_s
        discrete.DiscreteEnv.__init__(self, self.num_states, self.num_actions, self.P, self.initial_state_distrib)

    def create_empty_map(self, size):
        domain_map = []
        abs_size_x = size * 2 + 1
        abs_size_y = size + 2
        for i in range(abs_size_y):
            map_row = []
            for j in range(abs_size_x):
                if i == 0 or i == abs_size_y - 1:
                    if j == 0 or j == abs_size_x - 1:
                        map_row.append('+')
                    else:
                        map_row.append('-')
                elif j == 0 or j == abs_size_x - 1:
                    map_row.append('|')
                elif j % 2 == 1:
                    map_row.append(' ')
                else:
                    map_row.append(':')
            domain_map.append("".join(map_row))
        return domain_map

    def update_map(self, loc_row, loc_column, char):
        temp_row = list(self._str_map[loc_row])  # to list
        temp_row[loc_column] = char
        self._str_map[loc_row] = "".join(temp_row)
        self.desc = np.asarray(self._str_map, dtype='c')

    def get_dist_to_goal(self, d_r, d_c, p_r, p_c, t_r, t_c, is_pass_on_taxi):
        if is_pass_on_taxi:
            steps_to_complete = cityblock([t_r, t_c], [d_r, d_c])
        else:
            steps_to_complete = cityblock([t_r, t_c], [p_r, p_c]) + cityblock([p_r, p_c], [d_r, d_c])
        return steps_to_complete

    def is_possible_initial_state(self, pass_idx, dest_idx, taxi_row, taxi_col, fuel):
        d_r, d_c = self.destination_location[dest_idx]
        is_pass_on_taxi = False
        if pass_idx != len(self.passengers_locations):
            p_r, p_c = self.passengers_locations[pass_idx]
        else:
            p_r, p_c = taxi_row, taxi_col
            is_pass_on_taxi = True
        steps_to_complete = self.get_dist_to_goal(d_r, d_c, p_r, p_c, taxi_row, taxi_col, is_pass_on_taxi)
        cond = (pass_idx < len(self.passengers_locations) and fuel >= steps_to_complete and ((d_r != taxi_row or d_c != taxi_col)or pass_idx < len(self.passengers_locations)))
        # if cond == False:
        # print("DEBUG", pass_idx, pass_idx < len(self.passengers_locations), fuel >= steps_to_complete, ((d_r != taxi_row or d_c != taxi_col)
        # or pass_idx < len(self.passengers_locations)), fuel, steps_to_complete, d_r != taxi_row, d_c != taxi_col, taxi_row, taxi_col, d_r, d_c)

        return cond

    def get_info_from_map(self):
        fuel_station = None
        passengers_locations = []
        dest_locations = []
        h, w = self.desc.shape
        h = (h - 2)
        w = (w - 2)
        for x in range(1, h + 1):
            for y in range(1, w + 1):
                c = self.desc[x][y]
                if c == b'Y':
                    passengers_locations.append((x - 1, int(y / 2)))
                elif c == b'F':
                    fuel_station = (x - 1, int(y / 2))
                elif c == b'G':
                    dest_locations.append((x - 1, int(y / 2)))
        return passengers_locations, dest_locations, fuel_station

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx, fuel):
        # (5) 5, 5, 4, 50
        i = taxi_row
        i *= self.size
        i += taxi_col
        i *= len(self.passengers_locations) + 1
        i += pass_loc
        i *= len(self.destination_location)
        i += dest_idx
        i *= MAX_FUEL
        i += fuel

        return i

    def decode(self, i):
        # 50, 4, 5, 5, (5)
        out = []

        out.append(i % MAX_FUEL)  # fuel
        i = i // MAX_FUEL

        out.append(i % (len(self.destination_location)))  # dest
        i = i // (len(self.destination_location))

        out.append(i % (len(self.passengers_locations) + 1))  # loc
        i = i // (len(self.passengers_locations) + 1)
        out.append(i % self.size)  # cols

        i = i // self.size
        out.append(i)  # rows
        assert 0 <= i <= self.size
        return list(reversed(out))

    def get_map_image(self, adversarial=False):
        h, w = len(self._str_map), len(self._str_map[0])
        rgb_map = np.zeros((h, w, 3)).astype(np.uint8)
        if self.s != -1:
            taxi_row, taxi_col, pass_idx, dest_idx, fuel = self.decode(self.s)
        else:
            taxi_row, taxi_col, pass_idx, dest_idx, fuel = -1, -1, -1, -1, -1  # unintilized values
        taxi_status = "empty"
        if pass_idx == 1:  # len(self.passengers_locations):   SINGLE PASSANGER ONLY!
            taxi_status = "full"

        for r_index, row in enumerate(self._str_map):
            for c_index, char in enumerate(self._str_map[r_index]):
                cell_color = np.array(RGBCOLORS_MINUS_50.BLACK)  # BLACK-50
                if char == "+" or char == "-" or char == "|":
                    cell_color = np.array(RGBCOLORS_MINUS_50.GRAY)  # gray-50
                else:
                    map_taxi_row, map_taxi_col = self.translate_from_local_to_map(taxi_row, taxi_col)
                    # if taxi_col >= 0:
                    #     print("AAAAA", game_r, game_c, taxi_row, taxi_col)
                    num_objs_on_cell = 0
                    cell_color = np.array([0, 0, 0])

                    if r_index == map_taxi_row and c_index == map_taxi_col and taxi_status == "full":
                        num_objs_on_cell += 1
                        cell_color += RGBCOLORS_MINUS_50.GREEN
                    elif r_index == map_taxi_row and c_index == map_taxi_col and taxi_status == "empty":
                        num_objs_on_cell += 1
                        cell_color += RGBCOLORS_MINUS_50.YELLOW
                    if char == "G":
                        cell_color += RGBCOLORS_MINUS_50.MAGENTA
                        num_objs_on_cell += 1
                    elif char == "Y" and taxi_status == "empty":
                        cell_color += RGBCOLORS_MINUS_50.BLUE
                        num_objs_on_cell += 1
                    elif char == "F":
                        cell_color += RGBCOLORS_MINUS_50.RED
                        num_objs_on_cell += 1

                    if num_objs_on_cell > 0:
                        cell_color = (cell_color / num_objs_on_cell).astype(np.uint8)

                rgb_map[r_index, c_index] = cell_color
        if not self.fully_observed and not adversarial and taxi_row != -1:
            row, col = map_taxi_row, map_taxi_col
            min_row, max_row = max(row - self.agent_view_size, 0), min(row + self.agent_view_size, self.total_row_size)
            min_col, max_col = max(col - self.agent_view_size, 0), min(col + self.agent_view_size, self.total_col_size)
            rgb_map[min_row:max_row, min_col:max_col] = rgb_map[min_row:max_row, min_col:max_col] + 50
        else:
            rgb_map = rgb_map + 50

        return rgb_map

    def render(self, mode='human'):
        if mode == "rgb_array":
            return self.get_map_image()

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxi_row, taxi_col, pass_idx, dest_idx, fuel = self.decode(self.s)

        def ul(x):
            return "_" if x == " " else x

        if pass_idx < len(self.passengers_locations):
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(out[1 + taxi_row][2 * taxi_col + 1], 'yellow',
                                                                 highlight=True)

            # print(pass_idx)
            pi, pj = self.passengers_locations[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(ul(out[1 + taxi_row][2 * taxi_col + 1]), 'green',
                                                                 highlight=True)

        di, dj = self.destination_location[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
        else:
            outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    def sample_random_state(self, seed=None):
        # np.random.choice
        temp_s = self.s
        r_s = self.np_random.choice(self.all_legal_states_to_random_sample)
        self.s = r_s
        obs = self.get_observation()
        self.s = temp_s
        return obs

    def _seed(self, seed=None) -> list:
        """
        Setting a seed for the random sample state generation.
        Args:
            seed: seed to use

        Returns: list[seed]

        """
        self.np_random, self.seed_id = seeding.np_random(seed)
        return np.array([self.seed_id])

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

    def find_free_space_and_put(self, symbol):
        free_indices = [i for i in range(self.num_columns * self.num_rows) if self.get_map_symbol(self.unpack_index(i)[0], self.unpack_index(i)[1], False) == " "]
        index = np.random.choice(free_indices)
        loc_row, loc_col = self.unpack_index(index)
        self.put_in_map(loc_row, loc_col, symbol)

    def translate_from_map_to_local(self, row, col, wall_zone=False):
        offset_row = 1
        offset_col = 1
        if wall_zone:
            offset_col = 2

        new_col = col // 2
        new_row = row - offset_row
        assert new_col >= 0
        assert new_row >= 0
        return new_row, new_col

    def get_map_symbol(self, row, col, wall_zone=False):
        new_row, new_col = self.translate_from_local_to_map(row, col, wall_zone)

        return self._str_map[new_row][new_col]

    def translate_from_local_to_map(self, row, col, wall_zone=False):
        offset_row = 1
        offset_col = 1
        if wall_zone:
            offset_col = 2
        col = 2 * col
        return offset_row + row, offset_col + col

    def put_in_map(self, row, col, char, wall_zone=False):

        new_row, new_col = self.translate_from_local_to_map(row, col, wall_zone)
        self.update_map(new_row, new_col, char)

    def pack_index(self, row, col):
        loc = row * self.num_columns + col
        return loc

    def unpack_index(self, loc):
        col = int(loc % (self.num_columns))
        row = int(loc / (self.num_rows))
        return row, col

    def reset_agent(self):
        """Resets the agent's start position, but leaves goal and walls."""
        # Remove the previous agents from the world
        if self.adversary_step_count >= self.adversary_min_steps_for_init_map:
            self.adversary_step_done = True
            self.compute_shortest_path()
            self.init_after_adverserial()
        else:
            print("Error trying to reset agent before making adversarial step")

        self.step_count = 0
        self.s = self.init_s
        self.last_action = None

        # Return first observation
        return self.get_observation()

    def generate_random_z(self):
        return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)

    def reset(self):
        """Fully resets the environment to an empty grid with no agent or goal."""
        self.n_clutter_placed = 0
        self.adversary_step_count = 0
        self._str_map = self.create_empty_map(self.size)
        self.desc = np.asarray(self._str_map, dtype='c')

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

    # def reset(self):
        # self.s = self.init_s  # discrete.categorical_sample(self.isd, self.np_random)  # init at random state
    #     self.last_action = None
    #     return int(self.s)

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
        taxi_row, taxi_col, pass_idx, dest_idx, fuel = self.decode(self.s)

        if not self.fully_observed:
            agent_image = np.zeros((self.agent_view_size * 2, self.agent_view_size * 2, 3)).astype(np.uint8)
            gray = np.array([78] * 3) + 50  # gray - show walls in edges
            agent_image = agent_image + gray
            row, col = self.translate_from_local_to_map(taxi_row, taxi_col)

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
            'image': agent_image,
            'fuel': fuel
        }
        return obs

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
        self.last_action = a

        obs = self.get_observation()

        self.step_count += 1

        if self.step_count >= self.max_steps:
            done = True

        rewards = np.array([r])

        if self.minigrid_mode:
            rewards = rewards[0]
        return obs, rewards, done, {"prob": p}

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
        step_order = self.step_order  # ["choose_goal","choose_passanger", "choose_fuel", , "choose_agnet"]  # else choose_walls
        current_turn = step_order[self.adversary_step_count] if self.adversary_step_count < len(step_order) else "place_walls"
        if loc >= self.adversary_action_dim:
            raise ValueError('Position passed to step_adversary is outside the grid.')

        # Add offset of 1 for outside walls
        row, col = self.unpack_index(loc)

        done = False

        # Place goal
        if current_turn == "choose_goal":
            self.put_in_map(row, col, "G", False)  # Goal is "G"

        # Place the agent
        elif current_turn == "choose_passanger":
            # Goal has already been placed here
            if self.get_map_symbol(row, col, False) != " ":
                self.find_free_space_and_put("Y")
            else:
                self.put_in_map(row, col, "Y")  # passanger is "Y"
        elif current_turn == "choose_fuel":
            # Goal has already been placed here
            if self.get_map_symbol(row, col, False) != " ":
                self.find_free_space_and_put("F")
            else:
                self.put_in_map(row, col, "F")
        elif current_turn == "choose_agent":
            self.init_taxi_row, self.init_taxi_col = [row, col]
            self.deliberate_agent_placement = 1
        # Place wall
        elif self.adversary_step_count < self.adversary_max_steps:
            # If there is already an object there, action does nothing, also if it is on the grid bounderies
            if self.get_map_symbol(row, col, True) == ":":
                self.put_in_map(row, col, "|", wall_zone=True)
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
    env_id='MultiGrid-SingleTaxi-v0',
    entry_point=module_path + ':SingleTaxiEnv'
)


# def debug_print(l):
#     for x in l:
#         print(x)


# if __name__ == '__main__':
#     env = SingleTaxiEnv()
#     # debug_print(env._str_map)
#     for i in range(13):
#         env.step_adversary(2 * i)
#     # print(env.render())
#     # debug_print(env._str_map)

#     new_env = env
#     new_env.reset_agent()
#     new_env.render()
#     for _ in range(5):
#         next_s, r, done, prob = new_env.step(1)
#     new_env.render()
#     # print(new_env.s, r)
