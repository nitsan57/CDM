# -*- coding: utf-8 -*-

import gym
from gym.utils import seeding
import numpy as np
from social_rl.gym_multigrid.envs.taxi_config import TAXI_ENVIRONMENT_REWARDS, BASE_AVAILABLE_ACTIONS, ALL_ACTIONS_NAMES
from gym.spaces import Box, Tuple, MultiDiscrete
import random
import sys
from contextlib import closing
from io import StringIO
from enum import IntEnum
from gym import utils
import matplotlib.pyplot as plt
import os
os.system('')
from social_rl.gym_multigrid import register


orig_MAP = [
    "+---------+",
    "|X: |F: :X|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|X| :G|X: |",
    "+---------+",
]

EMPTY_MAP_SYMMETRIC = [
    "+---------+",
    "| : | : : |",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "| | : | : |",
    "+---------+",
]

# - 0: move south
# - 1: move north
# - 2: move east
# - 3: move west
# - 4: pickup passenger
# - 5: dropoff passenger
# - 6: turn engine on
# - 7: turn engine off
# - 8: standby
# - 9: refuel fuel tank

# Done completing task
# done = 6


class BCOLORS:
    RED = "\x1b[0;30;41m"
    GREEN = "\x1b[0;30;42m"
    YELLOW = "\x1b[0;30;43m"
    BLUE = "\x1b[0;30;44m"
    PINK = "\x1b[0;30;45m"
    CYAN = "\x1b[0;30;46m"
    #WHITE = "\x1b[0;30;47m"
    END = "\x1b[0m"


class RGBCOLORS_MINUS_50:
    BLACK = [0, 0, 0]  # -50
    GRAY = [78, 78, 78]  # -50
    RED = [205, 0, 0]  # -50
    GREEN = [0, 205, 0]  # -50
    YELLOW = [205, 205, 0]  # -50
    BLUE = [0, 94, 205]  #
    PINK = [205, 103, 104]  # -50
    CYAN = [0, 205, 205]  # -50


class TaxiEnv(gym.Env):
    """
    The Taxi Problem
    from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich
    Description:
    There are four designated locations in the grid world indicated by R(ed), G(reen), Y(ellow), and B(lue).
    When the episode starts, the taxi starts off at a random square and the passenger is at a random location.
    The taxi drives to the passenger's location, picks up the passenger, drives to the passenger's destination
    (another one of the four specified locations), and then drops off the passenger. Once the passenger is dropped off,
    the episode ends.

    Observations:
    A list (taxis, fuels, pass_start, destinations, pass_locs):
        taxis:                  a list of playable_coordinates of each taxi
        fuels:                  a list of fuels for each taxi
        pass_start:             a list of starting playable_coordinates for taeach passenger (current position or last available)
        destinations:           a list of destination playable_coordinates for each passenger
        passangers_locations:   a list of locations of each passenger.
                                -1 means delivered
                                0 means not picked up
                                positive number means the passenger is in the corresponding taxi number

    Passenger start: playable_coordinates of each of these
    - -1: In a taxi
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)

    Passenger location:
    - -1: delivered
    - 0: not in taxi
    - x: in taxi x (x is integer)

    Destinations: playable_coordinates of each of these
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)

    Fuel:
     - 0 to np.inf: default with 10

    Actions:
    Actions are given as a list, each element referring to one taxi's action. Each taxi has 7 actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: dropoff passenger
    - 6: turn engine on
    - 7: turn engine off
    - 8: standby
    - 9: refuel fuel tank


    Rewards:
    - Those are specified in the config file.

    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - other letters (R, G, Y and B): locations for passengers and destinations
    Main class to be characterized with hyper-parameters.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    # class Actions(IntEnum):
    #     # Turn left, turn right, move forward
    #     move_south = 0
    #     move_north = 1
    #     move_east = 2
    #     move_west = 3
    #     # Pick up an object
    #     pickup = 4
    #     # Drop an object
    #     drop = 5
    #     # Toggle/activate an object
    #     turn_engine_on = 6
    #     turn_engine_off = 7
    #     #
    #     standby = 8
    #     # refuel
    #     refuel = 9

    def __init__(self, _=0, n_clutter=10, size=5, agent_view_size=3, max_steps=250,
                 goal_noise=0., num_taxis: int = 1, num_passengers: int = 1, max_fuel: list = None,
                 domain_map: list = None, taxis_capacity: list = None, collision_sensitive_domain: bool = True,
                 fuel_type_list: list = None, option_to_stand_by: bool = False, random_z_dim=50):
        """
        TODO -  later version make number of passengers dynamic, even in runtime
        Args:
            num_taxis: number of taxis in the domain
            num_passengers: number of passengers occupying the domain at initiailaization
            max_fuel: list of max (start) fuel, we use np.inf as default for fuel free taxi.
            domain_map: 2D - map of the domain
            taxis_capacity: max capacity of passengers in each taxi (list)
            collision_sensitive_domain: is the domain show and react (true) to collisions or not (false)
            fuel_type_list: list of fuel types of each taxi
            option_to_stand_by: can taxis simply stand in place

            random_z_dim: The environment generates a random vector z to condition the

        """
        # Initializing default values

        ####PARAMS FOR ADVERSERIAL####
        self.minigrid_mode = True
        self.domain_map = self.create_empty_map(size=size)
        self.height = len(self.domain_map)
        self.width = len(self.domain_map[0])
        self.choose_goal_last = False
        self.max_steps = max_steps
        self.n_clutter = n_clutter
        self.n_agents = num_taxis
        self.adversary_max_steps = self.n_clutter + 2
        self.actions = BASE_AVAILABLE_ACTIONS
        self.random_z_dim = random_z_dim

        self.adversary_ts_obs_space = gym.spaces.Box(
            low=0, high=self.adversary_max_steps, shape=(1,), dtype='uint8')
        self.adversary_image_obs_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.height, self.width, 3),
            dtype='uint8')

        self.adversary_randomz_obs_space = gym.spaces.Box(low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)

        self.adversary_observation_space = gym.spaces.Dict(
            {'image': self.adversary_image_obs_space,
             'time_step': self.adversary_ts_obs_space,
             'random_z': self.adversary_randomz_obs_space})
        ##############################

        self.size = size
        self.agent_view_size = agent_view_size
        self.fully_observed = False
        self.num_taxis = num_taxis
        self.taxis_names = list(range(self.num_taxis))
        self.wall_locs = []
        self.step_order = ["choose_goal" * num_passengers, "choose_passanger" * num_passengers, "choose_fuel", "choose_gas", "choose_agent" * num_taxis]  # else choose_walls

        self.reset_agent_status()

        # Add four actions for placing the agent and goal and fuel and gas.

        self.num_taxis = num_taxis
        if max_fuel is None:
            self.max_fuel = [500] * num_taxis  # TODO - needs to figure out how to insert np.inf into discrete obs.space
        else:
            self.max_fuel = max_fuel

        if taxis_capacity is None:
            self.taxis_capacity = [1] * num_passengers
        else:
            self.taxis_capacity = taxis_capacity

        if fuel_type_list is None:
            self.fuel_type_list = ['F'] * num_passengers
        else:
            self.fuel_type_list = fuel_type_list

        self.init_map(self.domain_map, first_init=True)

        self.adversary_action_dim = self.num_rows * self.num_cols
        self.adversary_action_space = gym.spaces.Discrete(self.adversary_action_dim)

        self.collision_sensitive_domain = collision_sensitive_domain

        # Indicator list of 1's (collided) and 0's (not-collided) of all taxis
        self.collided = np.zeros(self.num_taxis)

        self.option_to_standby = option_to_stand_by

        # A list to indicate whether the engine of taxi i is on (1) or off (0), all taxis start as on.
        self.engine_status_list = list(np.ones(self.num_taxis).astype(bool))

        self.num_passengers = num_passengers

        # Available actions in relation to all actions based on environment parameters.
        self.available_actions_indexes, self.index_action_listionary, self.action_index_dictionary \
            = self._set_available_actions_dictionary()
        self.num_actions = len(self.available_actions_indexes)
        self.action_space = gym.spaces.Discrete(self.num_actions)
        if self.fully_observed:
            obs_image_shape = (self.height, self.width, 3)
        else:
            obs_image_shape = (self.agent_view_size * 2, self.agent_view_size * 2, 3)

        if self.minigrid_mode:
            msg = 'Backwards compatibility with minigrid only possible with 1 agent'
            assert self.n_agents == 1, msg

            # Single agent case
            # Actions are discrete integer values
            self.action_space = gym.spaces.Discrete(len(self.actions))
            # Images have three dimensions
            self.image_obs_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=obs_image_shape,
                dtype='uint8')
        else:
            # First dimension of all observations is the agent ID
            self.action_space = gym.spaces.Box(low=0, high=len(self.actions) - 1,
                                               shape=(self.n_agents,), dtype='int64')

            self.image_obs_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.n_agents,) + obs_image_shape,
                dtype='uint8')

        observation_space = {'image': self.image_obs_space}

        self.observation_space = gym.spaces.Dict(observation_space)

        self.last_action = None
        self.num_states = self._get_num_states()

        self._seed()

        self.np_random = None
        self.reset_agent(True)

        self.reset_metrics()

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

    def init_map(self, domain_map, first_init=False, adverserial_phase=False):
        #self.desc = np.asarray(domain_map, dtype='c')

        # Relevant features for map orientation, notice that we can only drive between the columns (':')
        self.num_rows = num_rows = len(domain_map) - 2
        self.num_cols = num_columns = len(domain_map[0][1:-1:2])

        # Set locations of passengers and fuel stations according to the map.

        self.passangers_locations = []
        self.fuel_station1 = None
        self.fuel_station2 = None
        self.fuel_stations = []
        if first_init != True:
            # initializing map with passengers and fuel stations
            for i, row in enumerate(domain_map[1:-1]):
                for j, char in enumerate(row[1:-1:2]):
                    loc = [i, j]
                    if char == 'X':
                        self.passangers_locations.append(loc)
                    elif char == 'F':
                        self.fuel_station1 = loc
                        self.fuel_stations.append(loc)
                    elif char == 'G':
                        self.fuel_station2 = loc
                        self.fuel_stations.append(loc)

            # self.playable_coordinates = [[i, j] for i in range(num_rows) for j in range(num_columns)]
            # print("PLAYABLE", self.playable_coordinates)
            # print("Call reset_agnet from map", adverserial_phase)
            self.reset_agent(in_init=adverserial_phase)

    def _get_num_states(self):
        map_dim = (self.num_rows * self.num_cols)
        passengers_loc_dim = 1
        for i in range(self.num_passengers):
            passengers_loc_dim *= len(self.passangers_locations) + self.num_taxis - i
        passengers_dest_dim = 1
        for i in range(self.num_passengers):
            passengers_dest_dim *= len(self.passangers_locations) - i
        num_states = map_dim * passengers_loc_dim * passengers_dest_dim
        return num_states

    def _get_observation_space_list(self) -> list:
        """
        Returns a list that emebed the observation space size in each dimension.
        An observation is a list of the form:
        [
            taxi_row, taxi_col, taxi_fuel,
            passenger1_row, passenger1_col,
            ...
            passenger_n_row, passenger_n_col,
            passenger1_dest_row, passenger1_dest_col,
            ...
            passenger_n_dest_row, passenger_n_dest_col,
            passenger1_status,
            ...
            passenger_n_status
        ]
        Returns: a list with all the dimensions sizes of the above.

        """
        locations_sizes = [self.num_rows, self.num_cols]
        fuel_size = [max(self.max_fuel) + 1]
        passengers_status_size = [self.num_taxis + 3]
        dimensions_sizes = []

        for _ in range(self.num_taxis):
            dimensions_sizes += locations_sizes
        for _ in range(self.num_taxis):
            dimensions_sizes += fuel_size

        for _ in range(self.num_passengers):
            dimensions_sizes += 2 * locations_sizes
        for _ in range(self.num_passengers):
            dimensions_sizes += passengers_status_size

        return [dimensions_sizes]

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
        taxis_locations, fuels, passangers_current_locations, destinations, passengers_status = self.state

        if len(destinations) == 0:
            return -1
        return int(destinations[0][0])

    def get_goal_y(self):
        taxis_locations, fuels, passangers_current_locations, destinations, passengers_status = self.state
        if len(destinations) == 0:
            return -1
        return int(destinations[0][1])

    def reset_metrics(self):
        self.distance_to_goal = -1
        self.n_clutter_placed = 0
        self.deliberate_agent_placement = -1
        self.passable = -1
        self.shortest_path_length = (self.num_cols) * (self.num_rows) + 1\


    def reset_agent_status(self):

        self.agent_starting_location = []
        self.passangers_start_locations = []
        taxis_locations, fuels, passangers_start_locations, passengers_destinations, passengers_status = [], [], [], [], []
        self.state = [taxis_locations, fuels, passangers_start_locations, passengers_destinations, passengers_status]
        self.dones = {taxi_index: False for taxi_index in self.taxis_names}
        self.dones['__all__'] = False

    def reset(self):
        """Fully resets the environment to an empty grid with no agent or goal."""
        self.domain_map = self.create_empty_map(size=self.size)

        self.init_map(self.domain_map, first_init=True)

        # Current position and direction of the agent
        self.reset_agent_status()

        # Extra metrics
        self.reset_metrics()

        image = self.get_map_image()
        self.step_count = 0
        self.adversary_step_count = 0
        obs = {
            'image': image,
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z()
        }

        return obs

    # def change_inner_state(self, new_state):
    #     self.state = new_state

    def sample_random_state(self, seed=None):

        # ["choose_goal","choose_passanger" * num_passengers, "choose_fuel", "choose_gas", , "choose_agnet" * num_taxis]  # else choose_walls
        # CONST OF MAP:
        # self.passengers_destinations

        # put the current agent in random position
        # chose location of agents

        taxis_locations, fuels, passangers_current_locations, destinations, passengers_status = self.state
        taxis_locations = []
        passangers_current_locations = []
        passengers_status = [2 for _ in range(self.num_passengers)]
        for a in range(self.n_agents):
            a_row_loc = np.random.randint(self.num_rows)
            a_col_loc = np.random.randint(self.num_cols)
            while [a_row_loc, a_col_loc] in taxis_locations:
                a_row_loc = np.random.randint(self.num_rows)
                a_col_loc = np.random.randint(self.num_cols)
            taxis_locations.append([a_row_loc, a_col_loc])
        # chose location for passangers:
        for p in range(self.num_passengers):
            p_row_loc = np.random.randint(self.num_rows)
            p_col_loc = np.random.randint(self.num_cols)
            while [p_row_loc, p_col_loc] in passangers_current_locations or [p_row_loc, p_col_loc] == destinations[p]:
                p_row_loc = np.random.randint(self.num_rows)
                p_col_loc = np.random.randint(self.num_cols)
            passangers_current_locations.append([p_row_loc, p_col_loc])
        for taxi_index, agent_loc in enumerate(taxis_locations):
            a_row, a_col = agent_loc[0], agent_loc[1]
            if [a_row, a_col] in passangers_current_locations:
                is_on_taxi = (np.random.rand() > 0.5)
                if is_on_taxi:
                    passanger_index = passangers_current_locations.index([a_row, a_col])
                    passengers_status[passanger_index] = taxi_index + 3

        state = taxis_locations, fuels, passangers_current_locations, destinations, passengers_status
        return state

    def reset_agent(self, in_init=False) -> dict:
        """
        Reset the environment's state:
            - taxis playable_coordinates.
            - refuel all taxis
            - random get destinations.
            - random locate passengers.
            - preserve other definitions of the environment (collision, capacity...)
            - all engines turn on.
        Args:

        Returns: The reset state.

        """
        # reset taxis locations
        taxis_locations = self.agent_starting_location
        self.collided = np.zeros(self.num_taxis)

        # refuel everybody
        fuels = [self.max_fuel[i] for i in range(self.num_taxis)]
        passangers_start_locations = []
        passengers_destinations = []
        if not in_init:
            passangers_start_locations = []
            for passanger in range(self.num_passengers):
                passangers_start_locations.append(self.passangers_locations[passanger])
            self.passangers_start_locations = passangers_start_locations
            self.passengers_destinations = passengers_destinations = self.passangers_locations[passanger + 1:]

        # Status of each passenger: delivered (1), in_taxi (positive number>2), waiting (2)
        passengers_status = [2 for _ in range(self.num_passengers)]
        self.state = [taxis_locations, fuels, passangers_start_locations, passengers_destinations, passengers_status]

        self.last_action = None
        # Turning all engines on
        self.engine_status_list = list(np.ones(self.num_taxis))

        # resetting dones
        self.dones = {taxi_id: False for taxi_id in self.taxis_names}
        self.dones['__all__'] = False

        self.step_count = 0

        obs = {}
        if not in_init:
            obs = self.get_observation()

        return obs

    def _set_available_actions_dictionary(self) -> (list, dict, dict):
        """

        TODO: Later versions - maybe return an action-dictionary for each taxi individually.

        Generates list of all available actions in the parametrized domain, index->action dictionary to decode.
        Generation is based on the hyper-parameters passed to __init__ + parameters defined in config.py

        Returns: list of available actions, index->action dictionary for all actions and the reversed dictionary
        (action -> index).

        """

        action_names = BASE_AVAILABLE_ACTIONS  # ALL_ACTIONS_NAMES  # From config.py
        base_dictionary = {}  # Total dictionary{index -> action_name}
        for index, action in enumerate(action_names):
            base_dictionary[index] = action

        available_action_list = BASE_AVAILABLE_ACTIONS  # From config.py

        if self.option_to_standby:
            available_action_list += ['turn_engine_on', 'turn_engine_off', 'standby']

        # TODO - when we return dictionary per taxi we can't longer assume that on np.inf fuel
        #  means no limited fuel for all the taxis
        # if not self.max_fuel[0] == np.inf:
        #     available_action_list.append('refuel')

        action_index_dictionary = dict((value, key) for key, value in base_dictionary.items())  # {action -> index} all
        available_actions_indexes = [action_index_dictionary[action] for action in available_action_list]
        index_action_listionary = dict((key, value) for key, value in base_dictionary.items())
        # print(list(set(available_actions_indexes)), index_action_listionary, action_index_dictionary)
        return list(set(available_actions_indexes)), index_action_listionary, action_index_dictionary

    def get_available_actions_dictionary(self) -> (list, dict):
        """
        Returns: list of available actions and index->action dictionary for all actions.

        """
        return self.available_actions_indexes, self.index_action_listionary

    def _is_there_place_on_taxi(self, passangers_locations: np.array, taxi_index: int) -> bool:
        """
        Checks if there is room for another passenger on taxi number 'taxi_index'.
        Args:
            passangers_locations: list of all passengers locations
            taxi_index: index of the desired taxi

        Returns: Whether there is a place (True) or not (False)

        """
        # Remember that passengers "location" is: 1 - delivered, 2 - waits for a taxi, >2 - on a taxi with index
        # location+2

        return (len([location for location in passangers_locations if location == (taxi_index + 3)]) <
                self.taxis_capacity[taxi_index])

    def map_at_location(self, location: list) -> str:
        """
        Returns the map character on the specified playable_coordinates of the grid.
        Args:
            location: location to check [row, col]

        Returns: character on specific location on the map

        """
        domain_map = self.domain_map.copy()
        row, col = location[0], location[1]
        return domain_map[row + 1][2 * col + 1]

    def at_valid_fuel_station(self, taxi: int, taxis_locations: list) -> bool:
        """
        Checks if the taxi's location is a suitable fuel station or not.
        Args:
            taxi: the index of the desired taxi
            taxis_locations: list of taxis playable_coordinates [row, col]
        Returns: whether the taxi is at a suitable fuel station (true) or not (false)

        """
        return (taxis_locations[taxi] in self.fuel_stations and
                self.map_at_location(taxis_locations[taxi]) == self.fuel_type_list[taxi])

    def _get_action_list(self, action_list) -> list:
        """
        Return a list in the correct format for the step function that should
        always get a list even if it's a single action.
        Args:
            action_list:

        Returns: list(action_list)

        """
        if type(action_list) == int:
            return [action_list]
        elif type(action_list) == np.int64:
            return [action_list]

        return action_list

    def _engine_is_off_actions(self, action: str, taxi: int) -> int:
        """
        Returns the reward according to the requested action given that the engine's is currently off.
        Also turns engine on if requested.
        Args:
            action: requested action
            taxi: index of the taxi specified, relevant for turning engine on
        Returns: correct reward

        """
        reward = self.partial_closest_path_reward('unrelated_action')
        if action == 'standby':  # standby while engine is off
            reward = self.partial_closest_path_reward('standby_engine_off')
        elif action == 'turn_engine_on':  # turn engine on
            reward = self.partial_closest_path_reward('turn_engine_on')
            self.engine_status_list[taxi] = 1

        return reward

    def _take_movement(self, action: str, row: int, col: int) -> (bool, int, int):
        """
        Takes a movement with regard to a apecific location of a taxi,
        Args:
            action: direction to move
            row: current row
            col: current col

        Returns: if moved (false if there is a wall), new row, new col

        """
        moved = False
        new_row, new_col = row, col
        max_row = self.num_rows - 1
        max_col = self.num_cols - 1
        if action == 'south':  # south
            if row != max_row:
                moved = True
            new_row = min(row + 1, max_row)
        elif action == 'north':  # north
            if row != 0:
                moved = True
            new_row = max(row - 1, 0)
        if action == 'east' and self.domain_map[1 + row][2 * col + 2] == ":":  # east
            if col != max_col:
                moved = True
            new_col = min(col + 1, max_col)
        elif action == 'west' and self.domain_map[1 + row][2 * col] == ":":  # west
            if col != 0:
                moved = True
            new_col = max(col - 1, 0)
        return moved, new_row, new_col

    def _check_action_for_collision(self, taxi_index: int, taxis_locations: list, current_row: int, current_col: int,
                                    moved: bool, current_action: int, current_reward: int) -> (int, bool, int, list):
        """
        Takes a desired location for a taxi and update it with regard to collision check.
        Args:
            taxi_index: index of the taxi
            taxis_locations: locations of all other taxis.
            current_row: of the taxi
            current_col: of the taxi
            moved: indicator variable
            current_action: the current action requested
            current_reward: the current reward (left unchanged if there is no collision)

        Returns: new_reward, new_moved, new_action_index

        """
        reward = current_reward
        row, col = current_row, current_col
        moved = moved
        action = current_action
        taxi = taxi_index
        # Check if the number of taxis on the destination location is greater than 0
        if len([i for i in range(self.num_taxis) if taxis_locations[i] == [row, col]]) > 0:
            if self.option_to_standby:
                moved = False
                action = self.action_index_dictionary['standby']
            else:
                self.collided[[i for i in range(len(taxis_locations)) if taxis_locations[i] == [row, col]]] = 1
                self.collided[taxi] = 1
                reward = self.partial_closest_path_reward('collision')
                taxis_locations[taxi] = [row, col]

        return reward, moved, action, taxis_locations

    def _make_pickup(self, taxi: int, passangers_current_locations: list, passengers_status: list,
                     taxi_location: list, reward: int) -> (list, int):
        """
        Make a pickup (successful or fail) for a given taxi.
        Args:
            taxi: index of the taxi
            passangers_current_locations: current locations of the passengers
            passengers_status: list of passengers statuses (1, 2, greater..)
            taxi_location: location of the taxi
            reward: current reward

        Returns: updates passengers status list, updates reward

        """
        passengers_status = passengers_status
        reward = reward
        successful_pickup = False
        for i, location in enumerate(passengers_status):
            # Check if we can take this passenger
            if location == 2 and taxi_location == passangers_current_locations[i] and \
                    self._is_there_place_on_taxi(passengers_status, taxi):
                passengers_status[i] = taxi + 3
                successful_pickup = True
                passangers_current_locations[i] = [-1, -1]
                reward = self.partial_closest_path_reward('pickup')
        if not successful_pickup:  # passenger not at location
            reward = self.partial_closest_path_reward('bad_pickup')

        return passengers_status, reward

    def _make_dropoff(self, taxi: int, current_passangers_current_locations: list, current_passengers_status: list,
                      destinations: list, taxi_location: list, reward: int) -> (list, list, int):
        """
        Make a dropoff (successful or fail) for a given taxi.
        Args:
            taxi: index of the taxi
            current_passangers_current_locations: current locations of the passengers
            current_passengers_status: list of passengers statuses (1, 2, greater..)
            destinations: list of passengers destinations
            taxi_location: location of the taxi
            reward: current reward

        Returns: updates passengers status list, updated passengers start location, updates reward

        """
        reward = reward
        passangers_current_locations = current_passangers_current_locations.copy()
        passengers_status = current_passengers_status.copy()
        successful_dropoff = False
        for i, location in enumerate(passengers_status):  # at destination
            location = passengers_status[i]
            # Check if we have the passenger and we are at his destination
            if location == (taxi + 3) and taxi_location == destinations[i]:
                passengers_status[i] = 1
                reward = self.partial_closest_path_reward('final_dropoff', taxi)
                passangers_current_locations[i] = taxi_location
                successful_dropoff = True
                break
            elif location == (taxi + 3):  # drops off passenger not at destination
                passengers_status[i] = 2
                successful_dropoff = True
                reward = self.partial_closest_path_reward('intermediate_dropoff', taxi)
                print("intermediate_dropoff reward", reward)
                passangers_current_locations[i] = taxi_location
                break
        if not successful_dropoff:  # not carrying a passenger
            reward = self.partial_closest_path_reward('bad_dropoff')

        return passengers_status, passangers_current_locations, reward

    def _update_movement_wrt_fuel(self, taxi: int, taxis_locations: list, wanted_row: int, wanted_col: int,
                                  reward: int, fuel: int) -> (int, int, list):
        """
        Given that a taxi would like to move - check the fuel accordingly and update reward and location.
        Args:
            taxi: index of the taxi
            taxis_locations: list of current locations (prior to movement)
            wanted_row: row after movement
            wanted_col: col after movement
            reward: current reward
            fuel: current fuel

        Returns: updated_reward, updated fuel, updared_taxis_locations

        """
        reward = reward
        fuel = fuel
        taxis_locations = taxis_locations
        if fuel == 0:
            reward = ('no_fuel')
        else:
            fuel = max(0, fuel - 1)

            taxis_locations[taxi] = [wanted_row, wanted_col]

        return reward, fuel, taxis_locations

    def _refuel_taxi(self, current_fuel: int, current_reward: int, taxi: int, taxis_locations: list) -> (int, int):
        """
        Try to refuel a taxi, if successful - updates fuel tank, if not - updates the reward.
        Args:
            current_fuel: current fuel of the taxi
            current_reward: current reward for the taxi.
            taxi: taxi index
            taxis_locations: list of current taxis locations

        Returns: updated reward, updated fuel

        """
        fuel = current_fuel
        reward = current_reward
        if self.at_valid_fuel_station(taxi, taxis_locations) and fuel != self.max_fuel[taxi]:
            fuel = self.max_fuel[taxi]
            reward += self.partial_closest_path_reward('refuel')
        else:
            reward = self.partial_closest_path_reward('bad_refuel')

        return reward, fuel

    def get_map_image(self):
        h, w = len(self.domain_map), len(self.domain_map[0])
        rgb_map = np.zeros((h, w, 3)).astype(np.uint8)

        for r_index, row in enumerate(self.domain_map):
            for c_index, char in enumerate(self.domain_map[r_index]):
                cell_color = np.array(RGBCOLORS_MINUS_50.BLACK)  # BLACK-50
                if char == "+" or char == "-" or char == "|":
                    cell_color = np.array(RGBCOLORS_MINUS_50.GRAY)  # gray-50
                objects = self.get_dynamic_map_objects(r_index, c_index, char)
                if len(objects) != 0:
                    cell_color = np.zeros(3).astype(np.float32)
                    if "full_taxi" in objects:
                        cell_color += np.array(RGBCOLORS_MINUS_50.GREEN).astype(np.float32)  # green-50
                    if "empty_taxi" in objects:
                        cell_color += np.array(RGBCOLORS_MINUS_50.YELLOW).astype(np.float32)
                    if "destination" in objects:
                        cell_color += np.array(RGBCOLORS_MINUS_50.PINK).astype(np.float32)
                    if "waiting_passanger" in objects:
                        cell_color += np.array(RGBCOLORS_MINUS_50.BLUE).astype(np.float32)
                    if "fuel_station" in objects:
                        cell_color += np.array(RGBCOLORS_MINUS_50.RED).astype(np.float32)
                    if "gas_station" in objects:
                        cell_color += np.array(RGBCOLORS_MINUS_50.CYAN).astype(np.float32)
                    cell_color = (cell_color / len(objects)).astype(np.uint8)
                rgb_map[r_index, c_index] = cell_color
        taxis_locations, fuels, passangers_current_locations, destinations, passengers_status = self.state
        if not self.fully_observed:
            for a in range(len(taxis_locations)):
                row, col = self.translate_from_local_to_map(taxis_locations[a][0], taxis_locations[a][1])
                min_row, max_row = max(row - self.agent_view_size, 0), min(row + self.agent_view_size, self.height)
                min_col, max_col = max(col - self.agent_view_size, 0), min(col + self.agent_view_size, self.width)
                rgb_map[min_row:max_row, min_col:max_col] = rgb_map[min_row:max_row, min_col:max_col] + 50
        else:
            rgb_map = rgb_map + 50

        return rgb_map

    def translate_from_local_to_map(self, row, col, wall_zone=False):
        offset_row = 1
        offset_col = 1
        if wall_zone:
            offset_col = 2

        col = 2 * col
        return offset_row + row, offset_col + col

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

        return self.domain_map[new_row][new_col]

    def get_dynamic_map_objects(self, row, col, char):

        taxis_locations, fuels, passangers_current_locations, destinations, passengers_status = self.state
        all_obj = []
        if row > 0 and col > 0:
            if char != ":" and char != "|":
                new_row, new_col = self.translate_from_map_to_local(row, col)
                try:
                    taxi_in_this_loc = taxis_locations.index([new_row, new_col])
                except:
                    taxi_in_this_loc = -1

                if taxi_in_this_loc != -1 and (taxi_in_this_loc + 3) in passengers_status:
                    all_obj.append("full_taxi")
                if [new_row, new_col] in taxis_locations:
                    all_obj.append("empty_taxi")
                if [new_row, new_col] in destinations:
                    all_obj.append("destination")
                if [new_row, new_col] in passangers_current_locations:
                    all_obj.append("waiting_passanger")
                if self.domain_map[row][col] == 'F':
                    all_obj.append("fuel_station")
                if self.domain_map[row][col] == 'G':
                    all_obj.append("gas_station")
        return all_obj

    def pretty_map_print(self):
        """    Rendering:
        - blue: passenger
        - magenta: destination
        - yellow: empty taxi
        - green: full taxi
        - other letters (R, G, Y and B): locations for passengers and destinations
        Main class to be characterized with hyper-parameters."""
        print(self.__str__())

    def put_in_map(self, row, col, char, wall_zone=False):

        new_row, new_col = self.translate_from_local_to_map(row, col, wall_zone)

        temp_row = list(self.domain_map[new_row])  # to list
        temp_row[new_col] = char
        self.domain_map[new_row] = "".join(temp_row)

    def pack_index(self, row, col):
        loc = row * self.num_cols + col
        return loc

    def unpack_index(self, loc):
        col = int(loc % (self.num_cols))
        row = int(loc / (self.num_rows))
        return row, col

    # def is_location_agnet_free(self, row,col):

    def find_free_space_and_put(self, symbol):
        free_indices = [i for i in range(self.num_cols * self.num_rows) if self.get_map_symbol(self.unpack_index(i)[0], self.unpack_index(i)[1], False) == " "]
        index = random.choice(free_indices)
        loc_row, loc_col = self.unpack_index(index)
        self.put_in_map(loc_row, loc_col, symbol)

    def generate_random_z(self):
        return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)

    def compute_shortest_path(self):
        "Currently supports single agent only"
        if len(self.agent_starting_location) == 0 or len(self.passangers_start_locations) == 0 or len(self.passengers_destinations) == 0:
            return

        self.distance_to_goal = abs(
            self.passengers_destinations[0][0] - self.passangers_start_locations[0][0]) + abs(
                self.passengers_destinations[0][1] - self.passangers_start_locations[0][1]) + abs(self.passangers_start_locations[0][0] - self.agent_starting_location[0][0]) + abs(self.passangers_start_locations[0][1] - self.agent_starting_location[0][1])

        taxis_locations, fuels, passangers_current_locations, destinations, passengers_status = self.state

        self.passable = -1

        wall_dict = {}

        # for a in range(self.n_agents)

        # row, col = self.translate_from_local_to_map(taxis_locations[0][0], taxis_locations[0][1])
        # d_r, d_c = destinations[0]

        # Check if there is a path between agent start position and goal. Remember
        # to subtract 1 due to outside walls existing in the Grid, but not in the
        # networkx graph.
        # self.passable=nx.has_path(
        #     self.graph,
        #     source = (self.agent_start_pos[0] - 1, self.agent_start_pos[1] - 1),
        #     target = (self.goal_pos[0] - 1, self.goal_pos[1] - 1))

        # if self.passable:
        #     # Compute shortest path
        #     self.shortest_path_length=nx.shortest_path_length(
        #         self.graph,
        #         source = (self.agent_start_pos[0] - 1, self.agent_start_pos[1] - 1),
        #         target = (self.goal_pos[0] - 1, self.goal_pos[1] - 1))
        # else:
        #     # Impassable environments have a shortest path length 1 longer than
        #     # longest possible path
        #     self.shortest_path_length=(self.width - 2) * (self.height - 2) + 1

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
        step_order = self.step_order  # ["choose_goal"* num_passengers,"choose_passanger" * num_passengers, "choose_fuel", "choose_gas", , "choose_agnet" * num_taxis]  # else choose_walls
        current_turn = step_order[self.adversary_step_count] if self.adversary_step_count < len(step_order) else "place_walls"
        if loc >= self.adversary_action_dim:
            raise ValueError('Position passed to step_adversary is outside the grid.')

        fuel_symbols = ["G", "F"]
        # agnet_banned_inital_symbols = ["X"]

        # Add offset of 1 for outside walls
        row, col = self.unpack_index(loc)

        done = False

        # Place goal
        if current_turn == "choose_goal":
            self.put_in_map(row, col, "X", False)  # passanger,and goal are "X"
        # Place the agent
        elif current_turn == "choose_passanger":
            # Goal has already been placed here
            if self.get_map_symbol(row, col, False) != " ":
                self.find_free_space_and_put("X")
            else:
                self.put_in_map(row, col, "X")  # passanger,and goal are "X"
        elif current_turn == "choose_fuel":
            # Goal has already been placed here
            if self.get_map_symbol(row, col, False) != " ":
                self.find_free_space_and_put("F")
            else:
                self.put_in_map(row, col, "F")
        elif current_turn == "choose_gas":
            # Goal has already been placed here
            if self.get_map_symbol(row, col, False) != " ":
                self.find_free_space_and_put("G")
            else:
                self.put_in_map(row, col, "G")
        elif current_turn == "choose_agent":
            # Goal has already been placed here
            # if self.get_map_symbol(row, col, False) != " ":
            #     # Place Passanger randomly
            #     free_indices = [i for i in range(self.num_cols * self.num_rows) if self.get_map_symbol(self.unpack_index(i)[0], self.unpack_index(i)[1], False) == " "]
            #     index = random.choice(free_indices)
            #     loc_row, loc_col = self.unpack_index(index)
            #     # agent is not represnted as part of the map
            #     self.agent_starting_location.append([loc_row, loc_col])
            #     self.deliberate_agent_placement = 0
            # else:
            # self.agent_starting_location.append([row, col])
            # self.deliberate_agent_placement = 1

            # we can put agent anywhere, but not where are other agents
            while [row, col] in self.agent_starting_location:
                row = np.random.randint(self.num_rows)
                col = np.random.randint(self.num_cols)
            else:
                self.agent_starting_location.append([row, col])
                self.deliberate_agent_placement = 1

        # Place wall
        elif self.adversary_step_count < self.adversary_max_steps:
            # If there is already an object there, action does nothing, also if it is on the grid bounderies
            if self.get_map_symbol(row, col, True) == ":":
                self.put_in_map(row, col, "|", wall_zone=True)
                self.n_clutter_placed += 1
                self.wall_locs.append((row, col))

                #self.wall_locs_dict[col] =  row
            # check if there are wall_barrier

        self.adversary_step_count += 1
        self.init_map(self.domain_map, first_init=False, adverserial_phase=True)
        # End of episode
        if self.adversary_step_count >= self.adversary_max_steps:
            done = True
            # Build graph after we are certain agent and goal are placed
            # for w in self.wall_locs:
            #     self.graph.remove_node(w)
            self.compute_shortest_path()

        image = self.get_map_image()
        obs = {
            'image': image,
            'time_step': [self.adversary_step_count],
            'random_z': self.generate_random_z()
        }

        return obs, 0, done, {}

    def step(self, action_list: list) -> (dict, dict, dict, dict):
        """
        Executing a list of actions (action for each taxi) at the domain current state.
        Supports not-joined actions, just pass 1 element instead of list.

        Args:
            action_list: {action} - action of specific taxis to take on the step

        Returns: - dict{taxi_id: observation}, dict{taxi_id: reward}, dict{taxi_id: done}, _
        """
        # taxis_locations, fuels, passangers_current_locations, destinations, passengers_status = self.state
        self.step_count += 1

        if isinstance(action_list, int):
            action_list = [action_list]

        rewards = [-1] * self.num_taxis

        _, index_action_listionary = self.get_available_actions_dictionary()

        # Main of the function, for each taxi-i act on action[i]
        for taxi_index, action in enumerate(action_list):
            # print("CHOSE ACTION", action_list, index_action_listionary[action])

            # meta operations on the type of the action
            # action = self._get_action_list(action_list)
            # print("XXXXXXXXXXXX", action)

            # for action in action_list:
            reward = self.partial_closest_path_reward('step')  # Default reward

            moved = False  # Indicator variable for later use

            taxis_locations, fuels, passangers_current_locations, destinations, passengers_status = self.state

            if all(list(self.dones.values())):
                rewards[taxi_index] = reward
                continue

            # If taxi is collided, it can't perform a step
            if self.collided[taxi_index] == 1:
                rewards[taxi_index] = self.partial_closest_path_reward('collided')
                self.dones[taxi_index] = True
                continue

            # If the taxi is out of fuel, it can't perform a step
            if fuels[taxi_index] == 0 and not self.at_valid_fuel_station(taxi_index, taxis_locations):
                rewards[taxi_index] = self.partial_closest_path_reward('bad_fuel')
                self.dones[taxi_index] = True
                continue

            taxi_location = taxis_locations[taxi_index]
            # print("TAXI LOCATION:", taxi_location)
            row, col = taxi_location

            fuel = fuels[taxi_index]
            is_taxi_engine_on = self.engine_status_list[taxi_index]

            if not is_taxi_engine_on:  # Engine is off
                # update reward according to standby/ turn-on/ unrelated + turn engine on if requsted
                reward = self._engine_is_off_actions(index_action_listionary[action], taxi_index)

            else:  # Engine is on
                # Binding
                if index_action_listionary[action] == 'bind':
                    reward = self.partial_closest_path_reward('bind')

                # Movement
                if index_action_listionary[action] in ['south', 'north', 'east', 'west']:
                    moved, row, col = self._take_movement(index_action_listionary[action], row, col)

                # Check for collisions
                if self.collision_sensitive_domain and moved:
                    if self.collided[taxi_index] == 0:
                        reward, moved, action, taxis_locations = self._check_action_for_collision(taxi_index,
                                                                                                  taxis_locations,
                                                                                                  row, col, moved,
                                                                                                  action, reward)

                # Pickup
                elif index_action_listionary[action] == 'pickup':
                    passengers_status, reward = self._make_pickup(taxi_index, passangers_current_locations,
                                                                  passengers_status, taxi_location, reward)

                # Dropoff
                elif index_action_listionary[action] == 'dropoff':
                    passengers_status, passangers_current_locations, reward = self._make_dropoff(taxi_index,
                                                                                                 passangers_current_locations,
                                                                                                 passengers_status,
                                                                                                 destinations,
                                                                                                 taxi_location,
                                                                                                 reward)

                # Turning engine off
                elif index_action_listionary[action] == 'turn_engine_off':
                    reward = self.partial_closest_path_reward('turn_engine_off')
                    self.engine_status_list[taxi_index] = 0

                # Standby with engine on
                elif index_action_listionary[action] == 'standby':
                    reward = self.partial_closest_path_reward('standby_engine_on')

            # Here we have finished checking for action for taxi-i
            # Fuel consumption
            if moved:
                reward, fuels[taxi_index], taxis_locations = self._update_movement_wrt_fuel(taxi_index, taxis_locations,
                                                                                            row, col, reward, fuel)

            if (not moved) and action in [self.action_index_dictionary[direction] for
                                          direction in ['north', 'south', 'west', 'east']]:
                reward = TAXI_ENVIRONMENT_REWARDS['hit_wall']

            # taxi refuel
            if index_action_listionary[action] == 'refuel':
                reward, fuels[taxi_index] = self._refuel_taxi(fuel, reward, taxi_index, taxis_locations)

            # check if all the passengers are at their destinations
            done = all(loc == 1 for loc in passengers_status)
            self.dones[taxi_index] = done

            # check if all taxis collided
            done = all(self.collided == 1)
            self.dones[taxi_index] = self.dones[taxi_index] or done

            # check if all taxis are out of fuel
            done = fuels[taxi_index] == 0
            self.dones[taxi_index] = self.dones[taxi_index] or done
            rewards[taxi_index] = reward
            self.state = [taxis_locations, fuels, passangers_current_locations, destinations, passengers_status]
            self.last_action = action_list

        self.dones['__all__'] = True
        self.dones['__all__'] = all(list(self.dones.values()))

        obs = {}
        # new_state = self.sample_random_state()
        # self.state = new_state
        obs = self.get_observation()

        # print("END OF STEP:", self.dones)
        collective_done = self.dones[0]  # SINGLE AGNET ASSUMPTION NITSAN

        if self.step_count >= self.max_steps:
            collective_done = True

        rewards = np.array([rewards[taxi_index] for taxi_index in range(len(action_list))])

        if self.minigrid_mode:
            rewards = rewards[0]
        # print("GOT REWARD FROM ACTION:", rewards, ALL_ACTIONS_NAMES[action_list[0]])
        return obs, rewards, collective_done, {}

    def render(self, mode: str = 'human') -> str:
        """
        Renders the domain map at the current state
        Args:
            mode: Demand mode (file or human watching).

        Returns: Value string of writing the output

        """
        if mode == "rgb_array":
            # self.pretty_map_print()
            # for debug only
            # if self.adversary_step_count >= 7:
            # obs = self.get_observation()['image']
            # plt.imshow(obs)
            # plt.show()
            return self.get_map_image()
        else:
            self.pretty_map_print()

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        # Copy map to work on
        out = self.domain_map.copy()
        #out = [[c.decode('utf-8') for c in line] for line in out]

        taxis, fuels, passengers_start_playable_coordinates, destinations, passangers_locations = self.state

        colors = ['yellow', 'red', 'white', 'green', 'cyan', 'crimson', 'gray', 'magenta'] * 5
        colored = [False] * self.num_taxis

        def ul(x):
            """returns underline instead of spaces when called"""
            return "_" if x == " " else x

        for i, location in enumerate(passangers_locations):
            if location > 2:  # Passenger is on a taxi
                taxi_row, taxi_col = taxis[location - 3]

                # Coloring taxi's coordinate on the map
                out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                    out[1 + taxi_row][2 * taxi_col + 1], colors[location - 3], highlight=True, bold=True)
                colored[location - 3] = True
            else:  # Passenger isn't in a taxi
                # Coloring passenger's playable_coordinates on the map
                pi, pj = passengers_start_playable_coordinates[i]
                out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)

        for i, taxi in enumerate(taxis):
            if self.collided[i] == 0:  # Taxi isn't collided
                taxi_row, taxi_col = taxi
                out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                    ul(out[1 + taxi_row][2 * taxi_col + 1]), colors[i], highlight=True)
            else:  # Collided!
                taxi_row, taxi_col = taxi
                out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                    ul(out[1 + taxi_row][2 * taxi_col + 1]), 'gray', highlight=True)

        for dest in destinations:
            di, dj = dest
            out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')
        outfile.write("\n".join(["".join(row) for row in out]) + "\n")

        if self.last_action is not None:
            moves = ALL_ACTIONS_NAMES
            output = [moves[i] for i in np.array(list(self.last_action.values())).reshape(-1)]
            outfile.write("  ({})\n".format(' ,'.join(output)))
        for i, taxi in enumerate(taxis):
            outfile.write("Taxi{}-{}: Fuel: {}, Location: ({},{}), Collided: {}\n".format(i + 1, colors[i].upper(),
                                                                                          fuels[i], taxi[0], taxi[1],
                                                                                          self.collided[i] == 1))
        for i, location in enumerate(passangers_locations):
            start = tuple(passengers_start_playable_coordinates[i])
            end = tuple(destinations[i])
            if location == 1:
                outfile.write("Passenger{}: Location: Arrived!, Destination: {}\n".format(i + 1, end))
            if location == 2:
                outfile.write("Passenger{}: Location: {}, Destination: {}\n".format(i + 1, start, end))
            else:
                outfile.write("Passenger{}: Location: Taxi{}, Destination: {}\n".format(i + 1, location - 2, end))
        outfile.write("Done: {}, {}\n".format(all(self.dones.values()), self.dones))
        outfile.write("Passengers Status's: {}\n".format(self.state[-1]))

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

    # @staticmethod
    # def partial_observations(state: list) -> list:
    #     """
    #     Get partial observation of state.
    #     Args:
    #         state: state of the domain (taxis, fuels, passengers_start_playable_coordinates, destinations, passangers_locations)

    #     Returns: list of observations s.t each taxi sees only itself

    #     """

    #     def flatten(x):
    #         return [item for sub in x for item in sub]

    #     observations = []
    #     taxis, fuels, passangers_current_locations, passengers_destinations, passangers_locations = state
    #     pass_info = flatten(passangers_current_locations) + flatten(passengers_destinations) + passangers_locations

    #     for i in range(len(taxis)):
    #         obs = taxis[i] + [fuels[i]] + pass_info
    #         obs = np.reshape(obs, [1, len(obs)])
    #         observations.append(obs)
    #     return observations

    def get_l1_distance(self, location1, location2):
        """
        Return the minimal travel length between 2 locations on the grid world.
        Args:
            location1: [i1, j1]
            location2: [i2, j2]

        Returns: np.abs(i1 - i2) + np.abs(j1 - j2)

        """
        return np.abs(location1[0] - location2[0]) + np.abs(location1[1] - location2[1])

    def get_observation(self) -> np.array:
        """
        Takes only the observation of the specified agent.
        Args:
            state: state of the domain (taxis, fuels, passengers_start_playable_coordinates, destinations, passangers_locations)
            agent_name: observer name
            agent_view_size: the size that the agent can see in the map (around it) in terms of other txis

        Returns: observation of the specified agent (state wise)

        """

        def flatten(x):
            return [item for sub in list(x) for item in list(sub)]

        images = []
        image = self.get_map_image()
        for a in range(self.n_agents):

            agent_image = np.copy(image)
            if not self.fully_observed:
                agent_image = np.zeros((self.agent_view_size * 2, self.agent_view_size * 2, 3)).astype(np.uint8)
                gray = np.array([78] * 3) + 50  # gray - show walls in edges
                agent_image = agent_image + gray
                taxis_locations, fuels, passangers_current_locations, destinations, passengers_status = self.state
                row, col = self.translate_from_local_to_map(taxis_locations[a][0], taxis_locations[a][1])

                min_row, max_row = max(row - self.agent_view_size, 0), min(row + self.agent_view_size, self.height - 1)
                min_col, max_col = max(col - self.agent_view_size, 0), min(col + self.agent_view_size, self.width - 1)
                sliced_image = image[min_row:max_row, min_col:max_col]
                start_row = 0
                end_row = self.agent_view_size * 2
                start_col = 0
                end_col = self.agent_view_size * 2
                if sliced_image.shape[0] != self.agent_view_size * 2:
                    if max_row == self.height - 1:
                        start_row = 0
                        end_row = sliced_image.shape[0]
                    if min_row == 0:
                        start_row = self.agent_view_size * 2 - sliced_image.shape[0]
                        end_row = self.agent_view_size * 2
                if sliced_image.shape[1] != self.agent_view_size * 2:
                    if max_col == self.width - 1:
                        start_col = 0
                        end_col = sliced_image.shape[1]
                    if min_col == 0:
                        start_col = self.agent_view_size * 2 - sliced_image.shape[1]
                        end_col = self.agent_view_size * 2
                agent_image[start_row: end_row, start_col:end_col] = sliced_image  # [start_row: end_row, start_col:end_col]

            images.append(agent_image)

        # Backwards compatibility: if there is a single agent do not return an array
        if self.minigrid_mode:
            images = images[0]

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # Note direction has shape (1,) for tfagents compatibility
        obs = {
            'image': images,
        }

        return obs

        # agent_index = self.taxis_names.index(agent_name)

        # taxis, fuels, passangers_current_locations, passengers_destinations, passangers_locations = state.copy()
        # passengers_information = flatten(passangers_current_locations) + flatten(
        #     passengers_destinations) + passangers_locations

        # closest_taxis_indices = []
        # for i in range(self.num_taxis):
        #     if self.get_l1_distance(taxis[agent_index], taxis[i]) <= self.agent_view_size and i != agent_index:
        #         closest_taxis_indices.append(i)

        # observations = taxis[agent_index].copy()
        # for i in closest_taxis_indices:
        #     observations += taxis[i]
        # observations += [0, 0] * (self.num_taxis - 1 - len(closest_taxis_indices)) + [fuels[agent_index]] + \
        #                 [0] * (self.num_taxis - 1) + passengers_information
        # observations = np.reshape(observations, (1, len(observations)))

        # return observations

    def passenger_destination_l1_distance(self, passenger_index, current_row: int, current_col: int) -> int:
        """
        Returns the manhattan distance between passenger current defined "start location" and it's destination.
        Args:
            passenger_index: index of the passenger.
            current_row: current row to calculate distance from destination
            current_col: current col to calculate distance from destination

        Returns: manhattan distance

        """
        current_state = self.state
        destination_row, destination_col = current_state[3][passenger_index]
        return int(np.abs(current_col - destination_col) + np.abs(current_row - destination_row))

    def partial_closest_path_reward(self, basic_reward_str: str, taxi_index: int = None) -> int:
        """
        Computes the reward for a taxi and it's defined by:
        dropoff[s] - gets the reward equal to the closest path multiply by 15, if the drive got a passenger further
        away - negative.
        other actions - basic reward from config table
        Args:
            basic_reward_str: the reward we would like to give
            taxi_index: index of the specific taxi

        Returns: updated reward

        """
        # print(basic_reward_str, TAXI_ENVIRONMENT_REWARDS[basic_reward_str])
        # True or taxi_index is None:  # basic_reward_str not in ['intermediate_dropoff', 'final_dropoff'] or taxi_index is None:
        if True or basic_reward_str not in ['intermediate_dropoff', 'final_dropoff'] or taxi_index is None:
            return TAXI_ENVIRONMENT_REWARDS[basic_reward_str]

        # [taxis_locations, fuels, passangers_current_locations, destinations, passengers_status]
        current_state = self.state
        passangers_current_locations = current_state[2]

        taxis_locations = current_state[0]

        passengers_status = current_state[-1]
        passenger_index = passengers_status.index(taxi_index + 3)
        passenger_start_row, passenger_start_col = passangers_current_locations[passenger_index]
        taxi_current_row, taxi_current_col = taxis_locations[taxi_index]

        return 15 * (self.passenger_destination_l1_distance(passenger_index, passenger_start_row, passenger_start_col) -
                     self.passenger_destination_l1_distance(passenger_index, taxi_current_row, taxi_current_col))

    def __str__(self):
        all_map = []
        line_list = []

        for row, line in enumerate(self.domain_map):
            for col, char in enumerate(line):
                if row > 0 and col > 0:
                    if char != ":" and char != "|":
                        objects = self.get_dynamic_map_objects(row, col, char)
                        if len(objects) != 0:
                            obj = objects[0]
                            if obj == "full_taxi":
                                line_list.append(BCOLORS.GREEN + char + BCOLORS.END)
                            elif obj == "empty_taxi":
                                line_list.append(BCOLORS.YELLOW + char + BCOLORS.END)
                            elif obj == "destination":
                                line_list.append(BCOLORS.PINK + char + BCOLORS.END)
                            elif obj == "waiting_passanger":
                                line_list.append(BCOLORS.BLUE + char + BCOLORS.END)
                            elif obj == "fuel_station":
                                line_list.append(BCOLORS.RED + char + BCOLORS.END)
                            elif obj == "gas_station":
                                line_list.append(BCOLORS.CYAN + char + BCOLORS.END)
                        else:
                            line_list.append(char)
                    else:
                        line_list.append(char)

                else:
                    line_list.append(char)
            line = "".join(line_list)
            all_map.append(line + '\n')
            # print(line)
            line_list = []
        return "".join(all_map)


if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname

register.register(
    env_id='MultiGrid-Taxi-Adversarial-v0',
    entry_point=module_path + ':TaxiEnv'
)
