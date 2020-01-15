import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from probecon.system_models.cart_pole import CartPole

class TrajectoryDataSet(Dataset):
        def __init__(self, file=None):
            if file is not None:
                with open(file, 'rb') as open_file:
                    self.trajectories = pickle.load(open_file)
            else:
                self.trajectories = []

        def __len__(self):
            return self.trajectories.__len__()

        def __getitem__(self, item):
            return self.trajectories[item]

        def add_trajectory(self, trajectory):
            self.trajectories.append(trajectory)



class TransitionDataSet(Dataset):
    def __init__(self, file=None, traj_data_set=None, batch_size=1, shuffle=False):
        if file is not None:
            with open(file, 'rb') as open_file:
                self.trajectories = pickle.load(open_file)
        else:
            self.trajectories = []
        if traj_data_set is not None:
            self.trajectories.update(traj_data_set.trajectories)

        self.transitions = {}

        for trajectory in self.trajectories:
            self.add_trajectory(trajectory)

        self.batch_size = batch_size
        self.shuffle = shuffle

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
        with open(file, 'wb') as open_file:
            pickle.dump(open_file, self.trajectories)

    def add_trajectory(self, trajectory):
        self.trajectories.append(trajectory)
        old_states = torch.tensor(trajectory['states'][:-1])
        controls = torch.tensor(trajectory['controls'])
        states = torch.tensor(trajectory['states'][1:])
        time = torch.tensor(trajectory['time'][:-1])
        time_steps = torch.tensor(trajectory['time'][1:]-trajectory['time'][:-1])
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
        old_state = torch.tensor(old_state)
        control = torch.tensor(control)
        state = torch.tensor(state)
        self.transitions['old_states'] = torch.cat((self.transitions['old_states'], old_state))
        self.transitions['controls'] = torch.cat((self.transitions['controls'], control))
        self.transitions['states'] = torch.cat((self.transitions['states'], state))
        self.transitions['time'] = torch.cat((self.transitions['time'], time))
        self.transitions['time_steps'] = torch.cat((self.transitions['time_steps'], time_step))

    def get_training_sample_continuous(self, item):
        old_state, control, state, time, time_step = self.__getitem__(item)
        rhs = (state-old_state)/time_step # right-hand side approximation of the ODE using Euler scheme
        return old_state, control, rhs

    def get_training_sample_discrete(self, item):
        old_state, control, state, time, time_step = self.__getitem__(item)
        return old_state, control, state

    def get_batches_discrete(self):
        return [(old_state, control, state)
                for old_state, control, state, time, time_step in self.dataloader()]

    def get_batches_continuous(self):
        return [(old_state, control, (state - old_state) / time_step)
                for old_state, control, state, time, time_step in self.dataloader()]
    def dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=self.shuffle)


if __name__ == '__main__':
    tset = TransitionDataSet()
    env = CartPole()
    for episode in range(1):
        env.reset()
        for step in range(100):
            env.random_step()
        tset.add_trajectory(env.trajectory)

    for x, u, dx in tset.get_batches_continuous():
        print(x, u, dx)

    for x, u, x_ in tset.get_batches_discrete():
        print(x, u, x_)