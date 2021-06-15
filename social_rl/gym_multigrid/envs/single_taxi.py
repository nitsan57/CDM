import sys
from contextlib import closing
from io import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

import os
os.system('')
from social_rl.gym_multigrid import register
MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| : : | : |",
    "|Y: :F| :B|",
    "+---------+",
]

SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF, REFUEL = 0, 1, 2, 3, 4, 5, 6
PASSENGER_IN_TAXI = -1
STEP_REWARD, PICKUP_REWARD, BAD_PICKUP_REWARD, DROPOFF_REWARD, BAD_DROPOFF_REWARD, REFUEL_REWARD, BAD_REFUEL_REWARD, NO_FUEL_REWARD = "step", "good_pickup", "bad_pickup", "good_dropoff", "bad_dropoff", "good_refuel", "bad_refuel", "no_fuel"
ACTIONS = [SOUTH, NORTH, EAST, WEST, PICKUP, DROPOFF, REFUEL]
MAX_FUEL = 50
REWARD_DICT = {STEP_REWARD: -1,
               PICKUP_REWARD: 0, BAD_PICKUP_REWARD: -10,
               DROPOFF_REWARD: 20, BAD_DROPOFF_REWARD: -10,
               REFUEL_REWARD: 10, BAD_REFUEL_REWARD: -10, NO_FUEL_REWARD: -100}


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
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi
    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
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
    - other letters (R, G, Y and B): locations for passengers and destinations
    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')
        self.last_action = None
        self.passengers_locations, self.fuel_station = self.get_info_from_map()
        self.taxi_fuel = MAX_FUEL

        self.num_states = 500 * MAX_FUEL
        self.num_rows = 5
        self.num_columns = 5
        self.max_row = self.num_rows - 1
        self.max_col = self.num_columns - 1
        self.initial_state_distrib = np.zeros(self.num_states)
        self.num_actions = len(ACTIONS)
        self.passenger_in_taxi = len(self.passengers_locations)
        self.P = {state: {action: [] for action in range(self.num_actions)} for state in range(self.num_states)}
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                for pass_idx in range(len(self.passengers_locations) + 1):  # +1 for being inside taxi
                    for dest_idx in range(len(self.passengers_locations)):
                        for fuel in range(self.taxi_fuel):
                            state = self.encode(row, col, pass_idx, dest_idx, fuel)
                            if self.is_possible_initial_state(pass_idx, dest_idx, row, col):
                                self.initial_state_distrib[state] += 1
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
                                if action == EAST and self.desc[1 + row, 2 * col + 2] == b":" and fuel != 0:
                                    new_col = min(col + 1, self.max_col)
                                    if new_row == col + 1:
                                        fuel -= 1
                                elif action == WEST and self.desc[1 + row, 2 * col] == b":" and fuel != 0:
                                    new_col = max(col - 1, 0)
                                    if new_row == col - 1:
                                        fuel -= 1
                                elif action == PICKUP:  # pickup
                                    if pass_idx < self.passenger_in_taxi and taxi_loc == self.passengers_locations[
                                            pass_idx]:
                                        new_pass_idx = self.passenger_in_taxi
                                    else:  # passenger not at location
                                        reward = REWARD_DICT[BAD_PICKUP_REWARD]
                                elif action == DROPOFF:  # dropoff
                                    if (taxi_loc == self.passengers_locations[
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
                                        fuel = MAX_FUEL
                                    else:
                                        reward = REWARD_DICT[BAD_REFUEL_REWARD]
                                elif fuel == 0:
                                    done = True
                                    reward = REWARD_DICT[NO_FUEL_REWARD]
                                new_state = self.encode(new_row, new_col, new_pass_idx, dest_idx, fuel)
                                self.P[state][action].append((1.0, new_state, reward, done))
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(self, self.num_states, self.num_actions, self.P, self.initial_state_distrib)

    def is_possible_initial_state(self, pass_idx, dest_idx, row, col):
        return pass_idx < 4 and pass_idx != dest_idx

    def get_info_from_map(self):
        fuel_station = None
        passenger_locations = []
        h, w = self.desc.shape
        h = (h - 2)
        w = (w - 2)
        for x in range(1, h + 1):
            for y in range(1, w + 1):
                c = self.desc[x][y]
                if c == b'R' or c == b'G' or c == b'B' or c == b'Y':
                    passenger_locations.append((x - 1, int(y / 2)))
                elif c == b'F':
                    fuel_station = (x - 1, int(y / 2))
        return passenger_locations, fuel_station

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx, fuel):
        # (5) 5, 5, 4, 50
        i = taxi_row
        i *= 5
        i += taxi_col
        i *= 5
        i += pass_loc
        i *= 4
        i += dest_idx
        i *= MAX_FUEL
        i += fuel
        return i

    def decode(self, i):
        # 50, 4, 5, 5, (5)
        out = []
        out.append(i % MAX_FUEL)
        i = i // 50
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        assert 0 <= i < 5
        return list(reversed(out))

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)

        def ul(x):
            return "_" if x == " " else x

        if pass_idx < 4:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(out[1 + taxi_row][2 * taxi_col + 1], 'yellow',
                                                                 highlight=True)
            pi, pj = self.passengers_locations[pass_idx]
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(ul(out[1 + taxi_row][2 * taxi_col + 1]), 'green',
                                                                 highlight=True)

        di, dj = self.passengers_locations[dest_idx]
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

    def reset(self):
        self.s = discrete.categorical_sample(self.isd, self.np_random)
        self.last_action = None
        return int(self.s)

    def step(self, a):
        transitions = self.P[self.s][a]
        i = discrete.categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.last_action = a
        return int(s), r, d, {"prob": p}


if hasattr(__loader__, 'name'):
    module_path = __loader__.name
elif hasattr(__loader__, 'fullname'):
    module_path = __loader__.fullname

register.register(
    env_id='MultiGrid-SingleTaxi-Adversarial-v0',
    entry_point=module_path + ':SingleTaxiEnv'
)
