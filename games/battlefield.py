import numpy
import random

import datetime
import os

# import numpy
from enum import Enum

import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (3, 10, 10)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(100))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 4  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 200  # Maximum number of moves if game is not finished before
        self.num_simulations = 20  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 6  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network

        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results",
                                         os.path.basename(__file__)[:-3], datetime.datetime.now().strftime(
                "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 1000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 128  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "SGD"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.002  # Initial learning rate
        self.lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        ### Replay Buffer
        self.replay_buffer_size = 3000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 32  # Number of game moves to keep for every batch element
        self.td_steps = 10  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = False  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Battlefield(seed)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """

        # print("step")
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """

        # print("legal_actions")
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        # input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.



        Returns:
            An integer from the action space.
        """

        player = "Computer"
        if self.env.player == 1:
            player = "User"
        # print(player + " step.")

        if self.env.stage == Stage_Arrangement:
            if self.env.ship_size is None:
                choice = input("Enter ship size(s: 1,2,3,4) and horizontal or vertical(p: 0,1), format: [sp]")
                return int(choice)
            else:
                choice = input("Enter ship position(0-99), format: [yx]")
                return int(choice)

        choice = input("Enter shooting coordinate(0-99), format: [yx]")
        return int(choice)
        # while choice not in [str(action) for action in self.legal_actions()]:
        #     choice = input("Enter the coordinate of field(from 0 to 99)")

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return f"{action_number}"

Board_Empty = 0
Board_Ship = 1
Board_Hit = 2
Board_Miss = 3

# game_count = 0
Stage_Arrangement = 0
Stage_Shooting = 1


class Battlefield:
    def __init__(self, seed):
        # print("init 1")
        self.random = numpy.random.RandomState(seed)
        self.player = 1
        # self.stage = Stage_Arrangement
        self.stage = Stage_Shooting
        # print("init 2")
        # self.user_board = Battlefield.get_battlefield()
        # self.comp_board = Battlefield.get_battlefield()
        self.user_board = self.place_ships(Battlefield.get_battlefield())
        self.comp_board = self.place_ships(Battlefield.get_battlefield())
        # print("init 3")

        self.player_ship_1 = 4
        self.player_ship_2 = 3
        self.player_ship_3 = 2
        self.player_ship_4 = 1

        self.comp_ship_1 = 4
        self.comp_ship_2 = 3
        self.comp_ship_3 = 2
        self.comp_ship_4 = 1

        self.player_ships = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]
        self.comp_ships = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]
        self.ship_size = None
        self.is_horizontal = None

        self.opponent_view = False
        # print("init")
        # self.render()

        # self.count_step = 0

    def legal_actions(self):
        # if self.stage == Stage_Arrangement:
        #     board = self.user_board if self.player == 1 else self.comp_board
        #     if self.ship_size:
        #         if self.is_horizontal:
        #             ori = "h"
        #         else:
        #             ori = "v"
        #         valid = False
        #         while (not valid):
        #             x = self.random.randint(1, 10) - 1
        #             y = self.random.randint(1, 10) - 1
        #             valid = validate(board, self.ship_size, x, y, ori)
        #         return x * 10 + y
        #     else:
        #         ships = self.player_ships if self.player == 1 else self.comp_ships
        #         ship = ships.pop()
        #         o = self.random.randint(0, 1)
        #         return ship * 10 + o

        board = self.comp_board if self.player == 1 else self.user_board
        points = []

        for i in range(10):
            for j in range(10):
                if board[i][j] == Board_Empty or board[i][j] == Board_Ship:
                    points.append(i * 10 + j)
        return points

    def reset(self):
        # global game_count
        # print("Reset game", game_count)
        # game_count = game_count + 1
        self.player = 1
        # self.user_board = Battlefield.get_battlefield()
        # self.comp_board = Battlefield.get_battlefield()
        self.user_board = self.place_ships(Battlefield.get_battlefield())
        self.comp_board = self.place_ships(Battlefield.get_battlefield())

        # self.stage = Stage_Arrangement
        self.stage = Stage_Shooting

        self.player_ship_1 = 4
        self.player_ship_2 = 3
        self.player_ship_3 = 2
        self.player_ship_4 = 1

        self.comp_ship_1 = 4
        self.comp_ship_2 = 3
        self.comp_ship_3 = 2
        self.comp_ship_4 = 1

        self.ship_size = None
        self.is_horizontal = None
        self.player_ships = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]
        self.comp_ships = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]

        # self.render()

        return self.get_observation()

    def place_ships(self, board):
        # types of ships
        ships = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4]

        # ship placement
        return self.computer_place_ships(board, ships)

    @staticmethod
    def get_battlefield():
        # setup blank 10x10 board
        # board = numpy.zeros((10, 10), dtype="int32")
        board = []
        for i in range(10):
            board_row = []
            for j in range(10):
                board_row.append(Board_Empty)
            board.append(board_row)
        return board

    def get_player_placed_ships(self):
        return self.player_ship_4 + self.player_ship_3 + self.player_ship_2 + self.player_ship_1

    def get_comp_placed_ships(self):
        return self.comp_ship_4 + self.comp_ship_3 + self.comp_ship_2 + self.comp_ship_1

    def set_ships(self, action):
        if action == 100:
            if self.player == 1:
                self.user_board = self.place_ships(self.user_board)
                self.player_ship_1 = 0
                self.player_ship_2 = 0
                self.player_ship_3 = 0
                self.player_ship_4 = 0

                self.player = 0
            else:
                self.comp_board = self.place_ships(self.comp_board)
                self.comp_ship_1 = 0
                self.comp_ship_2 = 0
                self.comp_ship_3 = 0
                self.comp_ship_4 = 0

                self.player = 1
            if self.get_player_placed_ships() == 0 and self.get_comp_placed_ships() == 0:
                self.stage = Stage_Shooting

            return True

        if self.ship_size is not None:
            x = int(action / 10)
            y = action % 10

            if self.is_horizontal:
                ori = "h"
            else:
                ori = "v"

            if self.player == 1:
                valid = validate(self.user_board, self.ship_size, x, y, ori)
                if valid:
                    self.user_board = place_ship(self.user_board, self.ship_size, ori, x, y)
                else:
                    return False
                if self.get_player_placed_ships() == 0:
                    self.player = 0
            else:
                valid = validate(self.comp_board, self.ship_size, x, y, ori)
                if valid:
                    self.comp_board = place_ship(self.comp_board, self.ship_size, ori, x, y)
                else:
                    return False
                if self.get_comp_placed_ships() == 0:
                    self.player = 1

            self.is_horizontal = None
            self.ship_size = None

            if self.get_player_placed_ships() == 0 and self.get_comp_placed_ships() == 0:
                self.stage = Stage_Shooting

        else:
            # xy
            # x - ship size from 1 to 4
            # y = horizontal/vertical 0 / 1

            x = int(action / 10)
            y = action % 10
            if y != 0 and y != 1:
                return False

            if self.player == 1:
                if x == 1 and self.player_ship_1 > 0:
                    self.player_ship_1 -= 1
                elif x == 2 and self.player_ship_2 > 0:
                    self.player_ship_2 -= 1
                elif x == 3 and self.player_ship_3 > 0:
                    self.player_ship_3 -= 1
                elif x == 4 and self.player_ship_4 > 0:
                    self.player_ship_4 -= 1
                else:
                    return False
            else:
                if x == 1 and self.comp_ship_1 > 0:
                    self.comp_ship_1 -= 1
                elif x == 2 and self.comp_ship_2 > 0:
                    self.comp_ship_2 -= 1
                elif x == 3 and self.comp_ship_3 > 0:
                    self.comp_ship_3 -= 1
                elif x == 4 and self.comp_ship_4 > 0:
                    self.comp_ship_4 -= 1
                else:
                    return False

            self.ship_size = x
            self.is_horizontal = y == 0
        return True

    def step(self, action):
        # print(str(self.player), str(action))
        if action == 101:
            self.opponent_view = True
            return self.get_observation(), 0, False

        # if self.stage == Stage_Arrangement:
        #     # self.count_step += 1
        #     reward = 1 if self.set_ships(action) else -1
        #     return self.get_observation(), reward, False

        # print("count steps: " + str(self.count_step))
        reward = 0
        x = int(action / 10)
        y = action % 10
        if self.player == 1:
            board = self.comp_board
        else:
            board = self.user_board

        res = make_move(board, x, y)
        if res == "hit":
            # print("Hit at " + str(x + 1) + "," + str(y + 1))
            board[x][y] = Board_Hit

            board, destroyed = self.check_ship_destroyed(board, x, y)
            reward = 1
            # reward = 2 if destroyed else 1
        elif res == "miss":
            # print("Sorry, " + str(x + 1) + "," + str(y + 1) + " is a miss.")
            board[x][y] = Board_Miss
            self.player = 0 if self.player == 1 else 1
        elif res == "try again":
            reward = -1

        # elif res == "try again":
        #     print("Sorry, that coordinate was already hit. Please try again")

        done = check_win(board)
        # reward = 1 if done else 0
        # print("check_win", done, self.get_reward(self.player))
        # if done:
        #     reward = 100
            # print("WIN:" + ("Player" if self.player == 1 else "MuZero"))
            # self.render()

        return self.get_observation(), reward, done

    def get_observation(self):
        if self.player == 1:
            user_board = self.user_board
            comp_board = hide_ships(self.comp_board)
        else:
            comp_board = self.comp_board
            user_board = hide_ships(self.user_board)

        # user_board = numpy.full((10, 10), user_board, dtype="float32")
        # comp_board = numpy.full((10, 10), comp_board, dtype="float32")
        board_to_play = numpy.full((10, 10), self.player, dtype="int32")

        return numpy.array([
            # numpy.full((10, 10), self.stage, dtype="float32"),
            user_board,
            comp_board,
            board_to_play,
        ])

    # def get_reward(self, player):
    #     if player == 1:
    #         board = self.comp_board
    #     else:
    #         board = self.user_board
    #
    #     destroyed_ships = 0
    #     for i in range(10):
    #         for j in range(10):
    #             if board[i][j] == Board_Hit:
    #                 destroyed_ships += 2
    #             if board[i][j] == Board_Ship:
    #                 destroyed_ships += 1
    #
    #     return destroyed_ships

    def check_ship_destroyed(self, board, x, y):
        top = self.get_top_ship(board, x, y)
        bottom = self.get_bottom_ship(board, x, y)
        left = self.get_left_ship(board, x, y)
        right = self.get_right_ship(board, x, y)

        if top is None or bottom is None or left is None or right is None:
            return board, False

        for i in range(max(x - left - 1, 0), min(x + right + 1 + 1, 10)):
            for j in range(max(y - top - 1, 0), min(y + bottom + 1 + 1, 10)):
                if board[i][j] == Board_Empty:
                    board[i][j] = Board_Miss
        return board, True

    def get_top_ship(self, board, x, y):
        if y == 0 or board[x][y - 1] == Board_Empty or board[x][y - 1] == Board_Miss:
            return 0
        if board[x][y - 1] == Board_Hit:
            top = self.get_top_ship(board, x, y - 1)
            if top is None:
                return None
            return 1 + top
        return None

    def get_bottom_ship(self, board, x, y):
        if y == 9 or board[x][y + 1] == Board_Empty or board[x][y + 1] == Board_Miss:
            return 0
        if board[x][y + 1] == Board_Hit:
            bottom = self.get_bottom_ship(board, x, y + 1)
            if bottom is None:
                return None
            return 1 + bottom
        return None

    def get_left_ship(self, board, x, y):
        if x == 0 or board[x - 1][y] == Board_Empty or board[x - 1][y] == Board_Miss:
            return 0
        if board[x - 1][y] == Board_Hit:
            top = self.get_left_ship(board, x - 1, y)
            if top is None:
                return None
            return 1 + top
        return None

    def get_right_ship(self, board, x, y):
        if x == 9 or board[x + 1][y] == Board_Empty or board[x + 1][y] == Board_Miss:
            return 0
        if board[x + 1][y] == Board_Hit:
            bottom = self.get_right_ship(board, x + 1, y)
            if bottom is None:
                return None
            return 1 + bottom
        return None

    # def get_near_ship_points(self, board, x, y):
    #     ship_points = [[x, y]]
    #
    #     if x > 0 and board[x - 1][y] == 2:
    #         ship_points.append([x - 1, y])
    #     if x < 9 and board[x + 1][y] == 2:
    #         ship_points.append([x + 1, y])
    #     if y > 0 and board[x][y - 1] == 2:
    #         ship_points.append([x, y - 1])
    #     if y < 9 and board[x][y + 1] == 2:
    #         ship_points.append([x, y + 1])
    #
    #     return ship_points;

    def to_play(self):
        return self.player

    def expert_action(self):
        board = self.user_board if self.player == 1 else self.comp_board

        if self.stage == Stage_Arrangement:
            if self.ship_size:
                if self.is_horizontal:
                    ori = "h"
                else:
                    ori = "v"
                valid = False
                while (not valid):
                    x = self.random.randint(1, 11) - 1
                    y = self.random.randint(1, 11) - 1
                    valid = validate(board, self.ship_size, x, y, ori)
                return x * 10 + y
            else:
                ships = self.player_ships if self.player == 1 else self.comp_ships
                ship = ships.pop()
                o = self.random.randint(0, 2)
                return ship * 10 + o

        # print_board("u", board, True)
        board = self.comp_board if self.player == 1 else self.user_board
        points = []

        for i in range(10):
            for j in range(10):
                if board[i][j] == Board_Empty or board[i][j] == Board_Ship:
                    points.append(i * 10 + j)
        next_point = self.random.choice(points)
        # print(str(points))
        # print("nextPoint")
        # print(str(next_point))

        return next_point

    def computer_place_ships(self, board, ships):
        for ship in ships:

            # generate random coordinates and validate the position
            valid = False
            while (not valid):
                x = self.random.randint(1, 11) - 1
                y = self.random.randint(1, 11) - 1
                o = self.random.randint(0, 2)
                # print(str(x) + " " + str(y) + " " + str(o))
                if o == 0:
                    ori = "v"
                else:
                    ori = "h"
                valid = validate(board, ship, x, y, ori)

            # place the ship
            # print("Computer placing a/an " + ship)
            board = place_ship(board, ship, ori, x, y)

        return board

    def render(self):
        print_board("u", self.user_board, self.opponent_view or self.player == 1)
        print_board("c", self.comp_board, self.opponent_view or self.player == 0)


def hide_ships(board):
    without_ships = []
    for i in range(10):
        board_row = []
        for j in range(10):
            if board[i][j] == Board_Ship:
                board_row.append(Board_Empty)
            else:
                board_row.append(board[i][j])
        without_ships.append(board_row)

    return without_ships


def print_board(s, board, show_ships):
    # WARNING: This function was crafted with a lot of attention. Please be aware that any
    #          modifications to this function will result in a poor output of the board
    #          layout. You have been warn.

    # find out if you are printing the computer or user board
    player = "Computer"
    if s == "u":
        player = "User"

    print("\nThe " + player + "'s board look like this:")

    # print the horizontal numbers
    line = "y x"
    for i in range(10):
        line += str(i) + " "
    print(line)

    for i in range(10):
        line = str(i) + "  "

        # print the board values, and cell dividers
        for j in range(10):
            if board[i][j] == Board_Empty:
                line += '0'
            elif board[i][j] == Board_Ship:
                line += "1" if show_ships else "0"
            elif board[i][j] == Board_Hit:
                line += "#"
            elif board[i][j] == Board_Miss:
                line += "*"
            else:
                line += " "

            if j != 9:
                line += " "

        print(line)


def place_ship(board, ship, ori, x, y):
    # place ship based on orientation
    if ori == "v":
        for i in range(ship):
            board[x + i][y] = Board_Ship
    elif ori == "h":
        for i in range(ship):
            board[x][y + i] = Board_Ship

    return board


def validate(board, ship, x, y, ori):
    # validate the ship can be placed at given coordinates
    if ori == "v" and x + ship > 10:
        return False
    elif ori == "h" and y + ship > 10:
        return False
    else:
        if ori == "v":
            for i in range(ship):
                if check_point(board, x + i, y):
                    return False
        elif ori == "h":
            for i in range(ship):
                if check_point(board, x, y + i):
                    return False

    return True


def check_point(board, x, y):
    for i in range(max(0, x - 1), min(x + 2, 10)):
        for j in range(max(0, y - 1), min(y + 2, 10)):
            if board[i][j] == Board_Ship:
                return True
    return False


def make_move(board, x, y):
    # make a move on the board and return the result, hit, miss or try again for repeat hit
    if board[x][y] == Board_Empty:
        return "miss"
    elif board[x][y] == Board_Ship:
        return "hit"
    else:
        return "try again"


def check_win(board):
    # simple for loop to check all cells in 2d board
    # if any cell contains a char that is not a hit or a miss return false
    for i in range(10):
        for j in range(10):
            if board[i][j] == Board_Ship:
                return False
    return True


# if __name__ == "__main__":
#     battlefield = Battlefield(1)
    # battlefield.step(99)
    # battlefield.step(98)
    # battlefield.step(97)
    # battlefield.step(96)
    # battlefield.step(95)
    # battlefield.step(94)
    # battlefield.step(93)
    # battlefield.step(92)


    # battlefield.render()
    # print("get_reward " + str(battlefield.get_reward(1)))
    # print("get_reward " + str(battlefield.get_reward(2)))
