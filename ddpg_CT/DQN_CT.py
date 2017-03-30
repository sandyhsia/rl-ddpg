import tensorflow as tf
import numpy as np
import random
from collections import deque
import time

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.05 # starting value of epsilon
FINAL_EPSILON = 0.05 # final value of epsilon
REPLAY_SIZE = 100000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
DEBUG_MODE = 0 # print some info and slow down.
USE_FC_ONLY = 1 # state is a 362-dim vector
USE_CONV = 0 # state is a 20x20-dim vector
USE_LSTM = 0 # adding rnn to the DQN
pack_size = 20 # pack one-dim vector into size*size*1 "img" so as to CONV

class DQN_CT():
        #DQN Continuous output for Agent
        def __init__(self, env):
                
                # ''env'' for car agent should be virtual sensor playground
                # env.observation should be 360 degree vec
                # env.action_space should be discrete action like 'wasd' x '0123' + 'q3' = 17

                # init experience replay
                self.replay_buffer = deque()
                # init some parameters
                self.time_step = 0
                self.epsilon = INITIAL_EPSILON

                
                self.state_dim = 362    # 360 vec + distance + angle diff
                self.action_dim = 2      # speed + steer
                self.speed_range = 5
                self.steer_range = 22.5
                self.session = tf.InteractiveSession()
                self.create_Q_network()
                self.create_training_method()

                self.session.run(tf.initialize_all_variables())
                self.time_t = 0
                self.train_time = 1
                self.loss = 0



        def create_Q_network(self):
                
                if USE_FC_ONLY == 1:
                        # network weights
                        # critic network Q(s, a)
                        self.s_W_fc1 = self.weight_variable([self.state_dim, 64])
                        self.s_b_fc1 = self.bias_variable([64])
                        self.a_W_fc1 = self.weight_variable([self.action_dim, 64])
                        self.a_b_fc1 = self.bias_variable([64])

                        self.W_fc2 = self.weight_variable([64, 20])
                        self.b_fc2 = self.bias_variable([20])

                        self.W_critic_Q = self.weight_variable([20, 1])
                        self.b_critic_Q = self.bias_variable([1])

                        #input layer
                        self.state_input = tf.placeholder("float", [None, self.state_dim])
                        self.action_input = tf.placeholder("float", [None, self.action_dim])
                        # hidden layers
                        c_h1_layer = tf.nn.relu(tf.matmul(self.state_input, self.s_W_fc1) + self.s_b_fc1 + tf.matmul(self.action_input, self.a_W_fc1) + self.a_b_fc1)
                        c_h2_layer = tf.nn.relu(tf.matmul(c_h1_layer, self.W_fc2) + self.b_fc2)
                        self.critic_Q_value = tf.matmul(c_h2_layer, self.W_critic_Q) + self.b_critic_Q
                        self.a_grads = tf.div(tf.gradients(self.critic_Q_value, self.action_input), tf.constant(BATCH_SIZE, dtype=tf.float32))


                        # actor network u(s)
                        # weight for policy output layer
                        self.s_W_fc2 =  self.weight_variable([self.state_dim, 64])
                        self.s_b_fc2 = self.bias_variable([64])
                        self.s_W_fc3 =  self.weight_variable([64, 20])
                        self.s_b_fc3 = self.bias_variable([20])
                        a_h1_layer = tf.nn.relu(tf.matmul(self.state_input, self.s_W_fc2) + self.s_b_fc2)
                        a_h2_layer = tf.nn.relu(tf.matmul(a_h1_layer, self.s_W_fc3) + self.s_b_fc3)

                        # weight for value output layer
                        self.W_fc4 = self.weight_variable([20, self.action_dim])
                        self.b_fc4 = self.bias_variable([self.action_dim])

                        self.actor_pi = tf.matmul(a_h2_layer, self.W_fc4)+self.b_fc4

                        # This gradient will be provided by the critic network
                        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim]) # feed in self.a_grads
                        # Combine the gradients here 
                        self.actor_gradients = tf.gradients(self.actor_pi, [self.s_W_fc2, self.s_b_fc2,  self.s_W_fc3, self.s_b_fc3, self.W_fc4, self.b_fc4], -self.action_gradient)



        def create_training_method(self):
                self.y_input = tf.placeholder("float", [None])
                # Q_action = tf.reduce_sum(tf.mul(self.Q_value, self.action_input), reduction_indices = 1)
                self.cost = tf.reduce_mean(tf.square(self.y_input - self.critic_Q_value))
                self.critic_optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)
                self.actor_optimizer = tf.train.AdamOptimizer(0.0001).\
                        apply_gradients(zip(self.actor_gradients, [self.s_W_fc2, self.s_b_fc2,  self.s_W_fc3, self.s_b_fc3, self.W_fc4, self.b_fc4]))


        def perceive(self, state, action, reward, next_state, done):
                self.time_t += 1
                
                if USE_FC_ONLY == 1:
                        self.replay_buffer.append((state, action, reward, next_state, done))

                if len(self.replay_buffer) > REPLAY_SIZE:
                        self.replay_buffer.popleft()

                if len(self.replay_buffer) > BATCH_SIZE*2 and self.time_t%self.train_time == 0:
                        self.train_Q_network()
                        self.time_t = 0

        def train_Q_network(self):

                self.time_step += 1
                # Step 1: obtain random minibatch from replay memory
                minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
                state_batch = [data[0] for data in minibatch] # refer to append order
                action_batch = [data[1] for data in minibatch]
                reward_batch = [data[2] for data in minibatch]
                next_state_batch = [data[3] for data in minibatch]

                if DEBUG_MODE:
                        print "Training. Please wait." 
                        print "Fetch minibatch[0] to check: ---------------"
                        print "state_batch: ", state_batch[0][360:362]
                        # print "action_batch: ", action_batch[0]
                        print "reward_batch: ", reward_batch[0]
                        # print "next_state_batch: ", next_state_batch[0]
                        print "done? : ", minibatch[0][4]
                        time.sleep(0.05)

                # Step 2: calculate y
                y_batch = []
                action_next_state_batch = self.actor_pi.eval(feed_dict={self.state_input: next_state_batch})
                Q_value_batch = self.critic_Q_value.eval(feed_dict={self.state_input: next_state_batch,
                                                                                                    self.action_input: action_next_state_batch})

                for i in range(0, BATCH_SIZE):
                        done = minibatch[i][4]
                        if done:
                                y_batch.append(reward_batch[i])
                        else:
                                y_batch.append(reward_batch[i]+GAMMA*np.max(Q_value_batch[i]))
                

                a_grads = self.a_grads.eval(feed_dict={
                            self.y_input: y_batch,
                            self.action_input: action_batch,
                            self.state_input: state_batch
                    })[0]

                self.critic_optimizer.run(feed_dict={
                            self.y_input: y_batch,
                            self.action_input: action_batch,
                            self.state_input: state_batch
                    })
                self.actor_optimizer.run(feed_dict={
                            self.state_input: state_batch,
                            self.action_gradient: a_grads
                    })

                loss = 0
                for i in range(0, BATCH_SIZE):
                        loss += (Q_value_batch[i][np.argmax(Q_value_batch[i])] - y_batch[i])**2
                self.loss += loss/BATCH_SIZE

                if DEBUG_MODE:
                        print "calculate y..." 
                        print "Q_value_batch: ", Q_value_batch[0]
                        print "y value: ", y_batch[0]
                        print "loss: ", loss/BATCH_SIZE
                        time.sleep(0.05)

        def egreedy_action(self, state):
                if USE_FC_ONLY == 1:
                        action = self.actor_pi.eval(feed_dict={
                            self.state_input: [state]
                            })[0] #?[0]

                        action = self.action_guard(action)
                        Q_value = self.critic_Q_value.eval(feed_dict = {
                            self.state_input:[state],
                            self.action_input: [action]
                            })[0]

                if DEBUG_MODE:
                        print "egreedy_action -- Q_value:", Q_value
                        print "agent speed", action[0]
                        print "agent steer", action[1]
                        time.sleep(0.05)


                if random.random() <= self.epsilon:
                        self.epsilon -= (self.epsilon - FINAL_EPSILON)/10000
                        ## Note!!!!
                        ## Initially, self.epsilon -= (INITIAL_EPSILON - FIANAL_EPSILON)/10000
                        ## But actually, result is after 10000/300 ~= 33 episode, exploration rate is 0.01
                        action = [0., 0.]
                        action = [random.random()*self.speed_range, random.random()*self.steer_range]
                        action = self.action_guard(action)
                        return action[0], action[1], Q_value
                
                else:
                        self.epsilon -= (self.epsilon - FINAL_EPSILON)/10000
                        return action[0], action[1], Q_value


        def weight_variable(self, shape):
                # initial = tf.truncated_normal(shape)
                return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

        def bias_variable(self, shape):
                initial = tf.constant(0.01, shape = shape)
                return tf.Variable(initial)

        def reset_loss(self):
                self.loss = 0
                return 1

        def get_loss(self):
                return self.loss

        def get_vars(self):
                return [self.s_W_fc1, self.s_b_fc1,
                            self.pi_W_fc1, self.pi_b_fc1,
                            self.W_fc2, self.b_fc2,
                            self.W_fc3, self.b_fc3,
                            self.W_fc4, self.b_fc4]
        
        def action_guard(self, action):
                
                if action[0] > self.speed_range:
                        action[0] = self.speed_range
                elif action[0] < -self.speed_range:
                        action[0] = -self.speed_range

                if action[1] > self.steer_range:
                        action[1] = self.steer_range
                elif action[1] < -self.steer_range:
                        action[1] = -self.steer_range

                return action



