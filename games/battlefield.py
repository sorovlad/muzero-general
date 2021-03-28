# import numpy
import random

import datetime
import os

# import numpy
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        self.observation_shape = (3, 3,
                                  3)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(2))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 4  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 21  # Maximum number of moves if game is not finished before
        self.num_simulations = 21  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 2  # Number of blocks in the ResNet
        self.channels = 32  # Number of channels in the ResNet
        self.reduced_channels_reward = 32  # Number of channels in reward head
        self.reduced_channels_value = 32  # Number of channels in value head
        self.reduced_channels_policy = 32  # Number of channels in policy head
        self.resnet_fc_reward_layers = [16]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [16]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [16]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = [16]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = [16]  # Define the hidden layers in the value network
        self.fc_policy_layers = [16]  # Define the hidden layers in the policy network

        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results",
                                         os.path.basename(__file__)[:-3], datetime.datetime.now().strftime(
                "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 15000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "SGD"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.03  # Initial learning rate
        self.lr_decay_rate = 0.75  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 150000

        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 20  # Number of game moves to keep for every batch element
        self.td_steps = 50  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
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
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
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
        observation, reward, done = self.env.step(action)
        return observation, reward * 10, done

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
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        choice = input(
            f"Enter the action (0) Hit, or (1) Stand for the player {self.to_play()}: "
        )
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter either (0) Hit or (1) Stand : ")
        return int(choice)

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return f"{action_number}"


class Battlefield:
    def __init__(self, seed):
        # self.random = numpy.random.RandomState(seed)
        self.player = 1
        self.user_board = self.get_battlefield()
        self.comp_board = self.get_battlefield()

    @staticmethod
    def get_battlefield():
        # types of ships
        ships = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4]

        # setup blank 10x10 board
        board = []
        for i in range(10):
            board_row = []
            for j in range(10):
                board_row.append(-1)
            board.append(board_row)

        # add ships as last element in the array
        board.append(ships)

        # ship placement
        return computer_place_ships(board, ships)

    @staticmethod
    def legal_actions():
        # 0 - 99 Координаты
        return list(range(100))

    def reset(self):
        self.player = 1
        self.user_board = self.get_battlefield()
        self.comp_board = self.get_battlefield()
        return self.get_observation()

    def step(self, action):
        x = int(action / 10)
        y = action % 10
        if self.player == 1:
            board = self.comp_board
        else:
            board = self.user_board

        res = make_move(board, x, y)
        if res == "hit":
            print("Hit at " + str(x + 1) + "," + str(y + 1))
            board[x][y] = 2
        elif res == "miss":
            print("Sorry, " + str(x + 1) + "," + str(y + 1) + " is a miss.")
            board[x][y] = 3
            if self.player == 1:
                self.player = 2
            else:
                self.player = 1

        elif res == "try again":
            print("Sorry, that coordinate was already hit. Please try again")

        done = check_win(board)

        return self.get_observation(), self.get_reward(self.player), done

    def get_observation(self):
        return [
            self.user_board,
            self.comp_board,
        ]

    def get_reward(self, player):
        if player == 1:
            board = self.comp_board
        else:
            board = self.user_board

        destroyed_ships = 0
        for i in range(10):
            for j in range(10):
                if board[i][j] == 2:
                    destroyed_ships += 1
        return destroyed_ships

    def to_play(self):
        return self.player

    def render(self):
        print_board("u", self.user_board)
        print_board("c", self.comp_board)


def print_board(s, board):
    # WARNING: This function was crafted with a lot of attention. Please be aware that any
    #          modifications to this function will result in a poor output of the board
    #          layout. You have been warn.

    # find out if you are printing the computer or user board
    player = "Computer"
    if s == "u":
        player = "User"

    print("The " + player + "'s board look like this: \n")

    # print the horizontal numbers
    line = "   "
    for i in range(10):
        line += str(i + 1) + "  "
    print(line)

    for i in range(10):
        line = ""
        # print the vertical line number
        if i != 9:
            line += str(i + 1) + "  "
        else:
            line += str(i + 1) + " "

        # print the board values, and cell dividers
        for j in range(10):
            if board[i][j] == -1:
                line += '0'
            elif board[i][j] == 1:
                line += "1"
            elif board[i][j] == 2:
                line += "#"
            elif board[i][j] == 3:
                line += "*"
            else:
                line += " "

            if j != 9:
                line += " "

        print(line)


def computer_place_ships(board, ships):
    for ship in ships:

        # generate random coordinates and validate the position
        valid = False
        while (not valid):
            x = random.randint(1, 10) - 1
            y = random.randint(1, 10) - 1
            o = random.randint(0, 1)
            if o == 0:
                ori = "v"
            else:
                ori = "h"
            valid = validate(board, ship, x, y, ori)

        # place the ship
        # print("Computer placing a/an " + ship)
        board = place_ship(board, ship, ori, x, y)

    return board


def place_ship(board, ship, ori, x, y):
    # place ship based on orientation
    if ori == "v":
        for i in range(ship):
            board[x + i][y] = 1
    elif ori == "h":
        for i in range(ship):
            board[x][y + i] = 1

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
            if board[i][j] == 1:
                print(str(i) + "  " + str(j))
                return True
    return False


def make_move(board, x, y):
    # make a move on the board and return the result, hit, miss or try again for repeat hit
    if board[x][y] == -1:
        return "miss"
    elif board[x][y] == '*' or board[x][y] == '$':
        return "try again"
    else:
        return "hit"


def check_win(board):
    # simple for loop to check all cells in 2d board
    # if any cell contains a char that is not a hit or a miss return false
    for i in range(10):
        for j in range(10):
            if board[i][j] != -1 and board[i][j] != '*' and board[i][j] != '$':
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
