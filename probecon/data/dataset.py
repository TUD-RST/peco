import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from probecon.system_models.cart_pole import CartPole

class TrajectoryDataSet(Dataset):
    """
    Data set for trajectories
    """
    def __init__(self, file=None):
        """

        Args:
            file (str):
                filename and path
        """
        if file is not None:
            with open(file, 'rb') as open_file:
                self.trajectories = pickle.load(open_file)
        else:
            self.trajectories = []

    def __len__(self):
        return self.trajectories.__len__()

    def __getitem__(self, item):
        """
        Returns a the trajectory with index 'item'

        Args:
            item (int):
                index of the trajectory that should be returned

        Returns:
            trajectory (dict):
                trajectory dictionary with the keys: 'old_states', 'controls', 'states', 'time' and 'time_steps'

        """
        trajectory = self.trajectories[item]
        return trajectory

    def add_trajectory(self, trajectory):
        """
        Add trajectory to the data set

        Args:
            trajectory (dict):
                trajectory dictionary with the keys: 'old_states', 'controls', 'states', 'time' and 'time_steps'
        """
        self.trajectories.append(trajectory)



class TransitionDataSet(Dataset):
    """
    Data set for transitions of dynamical systems. Can be used to train dynamics models.
    """
    def __init__(self, state_dim, control_dim,
                 file=None,
                 data_set=None,
                 batch_size=1,
                 shuffle=False,
                 type='continuous',
                 state_eq=None,
                 second_order=False):
        """

        Args:
            state_dim (int):
                state dimension
            control_dim (int):
                control input dimension
            file (str):
                file to load
            data_set (probecon.data.dataset.TrajectoryDataSet, probecon.data.dataset.TransitionDataSet):
                data set that is loaded
            batch_size (int):
                batch size for training
            shuffle (bool):
                if 'True', data is reshuffled at every epoch
            type (str):
                'continuous':
                    for training an ODE
                'discrete':
                    for training a difference equation
            state_eq (function):
                state equation which has to be an ODE of the form 'state_eq(time=0., state, control)'
            second_order (bool):
                if 'True', only half of the model is learned

        """
        if file is not None:
            with open(file, 'rb') as open_file:
                self.trajectories = pickle.load(open_file)
        else:
            self.trajectories = []
        if data_set is not None:
            self.trajectories.update(data_set.trajectories)

        self.transitions = {}

        for trajectory in self.trajectories:
            self.add_trajectory(trajectory)

        self.type = type
        self.batch_size = batch_size
        self.shuffle = shuffle
        if state_eq is not None:
            self.state_eq = state_eq
        else:
            raise ValueError
        self.state_dim = state_dim
        self.control_dim = control_dim
        if second_order and state_dim % 2 == 0:
            self.output_dim = int(state_dim/2)
        else:
            self.output_dim = state_dim

    def __len__(self):
        return self.transitions['time'].__len__()

    def __getitem__(self, item):
        old_state = self.transitions['old_states'][item]
        control = self.transitions['controls'][item]
        state = self.transitions['states'][item]
        time = self.transitions['time'][item]
        time_step = self.transitions['time_steps'][item]
        return old_state, control, state, time, time_step

    def save_to_file(self, file):
        """
        Save data set to file

        Args:
            file (str):
                file name and path

        """
        with open(file, 'wb') as open_file:
            pickle.dump(open_file, self.trajectories)
        pass

    def add_trajectory(self, trajectory):
        """
        Add trajectory to the data set

        Args:
            trajectory (dict):
                trajectory dictionary with the keys: 'old_states', 'controls', 'states', 'time' and 'time_steps'
        """
        self.trajectories.append(trajectory)
        old_states = torch.tensor(trajectory['states'][:-1], dtype=torch.float32)
        controls = torch.tensor(trajectory['controls'], dtype=torch.float32)
        states = torch.tensor(trajectory['states'][1:], dtype=torch.float32)
        time = torch.tensor(trajectory['time'][:-1], dtype=torch.float32)
        time_steps = torch.tensor(trajectory['time'][1:]-trajectory['time'][:-1], dtype=torch.float32)
        if list(self.transitions.keys())==[]:
            self.transitions['old_states'] = old_states
            self.transitions['controls'] = controls
            self.transitions['states'] = states
            self.transitions['time'] = time
            self.transitions['time_steps'] = time_steps
        else:
            self.transitions['old_states'] = torch.cat((self.transitions['old_states'], old_states))
            self.transitions['controls'] = torch.cat((self.transitions['controls'], controls))
            self.transitions['states'] = torch.cat((self.transitions['states'], states))
            self.transitions['time'] = torch.cat((self.transitions['time'], time))
            self.transitions['time_steps'] = torch.cat((self.transitions['time_steps'], time_steps))
        pass

    def add_transition(self, old_state, control, state, time, time_step):
        """
        Add a state transition to the data set.
        Args:
            old_state (numpy.ndarray):
                previous state of the environment
            control (numpy.ndarray):
                control input applied to the environment
            state (numpy.ndarray):
                state
            time (float):
                time when the transition happened
            time_step (float):
                time diffrence between 'old_state' and 'state'

        """
        old_state = torch.tensor(old_state)
        control = torch.tensor(control)
        state = torch.tensor(state)
        self.transitions['old_states'] = torch.cat((self.transitions['old_states'], old_state))
        self.transitions['controls'] = torch.cat((self.transitions['controls'], control))
        self.transitions['states'] = torch.cat((self.transitions['states'], state))
        self.transitions['time'] = torch.cat((self.transitions['time'], time))
        self.transitions['time_steps'] = torch.cat((self.transitions['time_steps'], time_step))
        pass

    def get_training_sample_continuous(self, item):
        """
        Create a training sample consisting of state, control and approximate state derivative

        Args:
            item (int):
                index of the training sample in the data set

        Returns:
            old_state (numpy.ndarray):
                state of the environment before applying the control input
            control (numpy.ndarray):
                control input
            state_diff (numpy.ndarray):
                approximate state derivative using an Euler forward scheme

        """
        old_state, control, state, time, time_step = self.__getitem__(item)
        state_diff = (state-old_state)/time_step # right-hand side approximation of the ODE using Euler scheme
        return old_state, control, state_diff

    def get_training_sample_discrete(self, item):
        """
        Create a training sample consisting of state, control and approximated state derivative

        Args:
            item (int):
                index of the training sample in the data set

        Returns:
            old_state (numpy.ndarray):
                state of the environment before applying the control input
            control (numpy.ndarray):
                control input applied to the environment
            state (numpy.ndarray):
                state of the environment before applying the control input

        """
        old_state, control, state, time, time_step = self.__getitem__(item)
        return old_state, control, state

    def get_batches(self):
        """
        Create batches from the data set

        Returns:
            batches (list):
                list of batches that contain a tuple (x, y), where x is the neural network input and y is the training target

        """
        if self.type=='continuous':
                batches = [(torch.cat((old_state, control), 1),
                            ((state - old_state) / time_step
                             - self.batch_state_eq(old_state, control))[:, self.state_dim-self.output_dim:])
                           for old_state, control, state, time, time_step in self.data_loader()]

        elif self.type=='discrete':
                batches = [(torch.cat((old_state, control), 1),
                            (state - old_state
                             - time_step*self.batch_state_eq(old_state, control))[:, self.state_dim-self.output_dim:])
                           for old_state, control, state, time, time_step in self.data_loader()]
        else:
            raise ValueError
        return batches

    def batch_state_eq(self, states, controls):
        """
        Create a batch of state equation outputs

        Args:
            states (numpy.ndarray):
                batch of states
            controls (numpy.ndarray):
                batch of control inputs

        Returns:
            y (torch.Tensor):
                batch of outputs of the state equation

        """
        y = torch.tensor([self.state_eq(0., state, control) for (state, control) in zip(states, controls)],
                         dtype=torch.float32)
        return y

    def data_loader(self):
        """
        Creates a the DataLoader object for iterating over the data set

        Returns:
            loader (torch.utils.data.DataLoader):
                provides an iterable over the data set

        """
        loader = DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)
        return loader


if __name__ == '__main__':
    tset = TransitionDataSet()
    env = CartPole()
    for episode in range(1):
        env.reset()
        for step in range(100):
            env.random_step()
        tset.add_trajectory(env.trajectory)

