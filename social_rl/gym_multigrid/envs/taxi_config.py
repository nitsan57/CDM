# TAXI_ENVIRONMENT_REWARDS = dict(
#     step=-1,
#     no_fuel=-20,
#     bad_pickup=-10,
#     bad_dropoff=-10,
#     bad_refuel=-10,
#     bad_fuel=-50,
#     pickup=10,
#     standby_engine_off=-1,
#     turn_engine_on=-1000,  # 10e6
#     turn_engine_off=-1000,  # 10e6
#     standby_engine_on=-1,
#     intermediate_dropoff=20,
#     final_dropoff=100,
#     hit_wall=-2,
#     collision=-35,
#     collided=-20,
#     unrelated_action=-15,
# )
TAXI_ENVIRONMENT_REWARDS = dict(
    step=-1,
    no_fuel=-100,
    bad_pickup=-10,
    bad_dropoff=-10,
    bad_refuel=-10,
    refuel=10,
    bad_fuel=-50,
    pickup=10,
    # standby_engine_off=-1,
    # turn_engine_on=-1000,  # 10e6
    # turn_engine_off=-1000,  # 10e6
    # standby_engine_on=-1,
    intermediate_dropoff=20,
    final_dropoff=100,
    hit_wall=0,
    # collision=-35,
    # collided=-20,
    # unrelated_action=-15,
)

COLOR_MAP = {
    ' ': [0, 0, 102],  # Black background
    '_': [0, 0, 102],
    '0': [0, 0, 102],  # Black background beyond map walls
    '': [180, 180, 180],  # Grey board walls
    '|': [180, 180, 180],  # Grey board walls
    '+': [180, 180, 180],  # Grey board walls
    '-': [180, 180, 180],  # Grey board walls
    ':': [0, 0, 102],  # black passes board walls
    '@': [180, 180, 180],  # Grey board walls
    'P': [254, 151, 0],  # [254, 151, 0],  # Blue
    'P0': [254, 151, 0],  # [102, 51, 0],
    'P1': [254, 151, 0],  # [153, 76, 0],
    'P2': [254, 151, 0],  # [204, 102, 0],
    'P3': [254, 151, 0],  # [255, 128, 0],
    'P4': [254, 151, 0],  # [255, 153, 51],
    'D': [102, 0, 51],
    'D0': [102, 0, 51],
    'D1': [102, 0, 51],  # [153, 0, 76],
    'D2': [102, 0, 51],  # [204, 0, 102],
    'D3': [102, 0, 51],  # [255, 0, 127],
    'D4': [102, 0, 51],  # [255, 51, 153],
    'F': [250, 204, 255],  # Pink
    'G': [159, 67, 255],  # Purple
    'X': [0, 0, 102],

    # Colours for agents. R value is a unique identifier
    '1': [255, 255, 000],  # Yellow
    '2': [255, 000, 000],  # Red
    '3': [204, 204, 204],  # White
    '4': [51, 255, 000],  # Green
    '5': [100, 255, 255],  # Cyan
}

ALL_ACTIONS_NAMES = ['south', 'north', 'east', 'west',
                     'pickup', 'dropoff', 'refuel', 'turn_engine_on', 'turn_engine_off',
                     'standby',
                     'refuel']

""",
                    'turn_engine_on', 'turn_engine_off',
                    'standby',
                    'refuel']"""

BASE_AVAILABLE_ACTIONS = ['south', 'north', 'east', 'west',
                          'pickup', 'dropoff', 'refuel']
