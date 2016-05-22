import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q = {}
        self.learning_rate = 0.3
        self.discount_factor = 0.2
        self.past_state = None
        self.past_reward = 0.0
        self.past_action = None

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required


    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        state =(self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'])
        self.state = state

        # If (state, action) not in Q, add it.
        for x in self.env.valid_actions:
            if (state, x) not in self.Q:
                self.Q[(state, x)] = 1.0  # random.random()

        # TODO: Select action according to your policy
        Q_action = []
        for x in self.env.valid_actions:
            Q_action.append(self.Q[(state, x)])
        # epsilon-greedy method
        epsilon = 0.1
        if random.random() < epsilon:
            action = random.choice(self.env.valid_actions)  # random move
        elif Q_action.count(max(Q_action)) > 1:
            # if more than one max value, pick one randomly
            tmp = np.array(Q_action)
            ind = np.where(tmp==max(tmp))
            action =self.env.valid_actions[random.choice(ind[0])]
        else:
            action = self.env.valid_actions[np.argmax(Q_action)]  # pick the best action from current state-based Q-values

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Get new sense and state
        new_inputs = self.env.sense(self)
        new_state = (self.planner.next_waypoint(), new_inputs['light'], new_inputs['oncoming'], new_inputs['left'])

        # If (state, action) not in Q, add it.
        for x in self.env.valid_actions:
            if (new_state, x) not in self.Q:
                self.Q[(new_state, x)] = 1.0  #random.random()

        # Calcuate the new Q value
        new_Q_action = []
        for x in self.env.valid_actions:
            new_Q_action.append(self.Q[(new_state, x)])

        # TODO: Learn policy based on state, action, reward
        if self.state != None:
            self.Q[(state, action)] = (1 - self.learning_rate) * self.Q[(state, action)] + \
            self.learning_rate * (reward + self.discount_factor * max(new_Q_action))
        self.state = new_state
        self.action =

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
