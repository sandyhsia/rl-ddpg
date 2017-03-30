""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import random
from collections import deque
import time

# ==========================
#   Training Parameters
# ==========================
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor 
GAMMA = 0.99
# Soft target update param
TAU = 0.01
INITIAL_EPSILON = 0.1 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 100000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
DEBUG_MODE = 0 # print some info and slow down.

def weight_variable(shape):
        # initial = tf.truncated_normal(shape)
        return tf.Variable(tf.truncated_normal(shape, stddev = 0.01))

def bias_variable(shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

# ===========================
#   Actor and Critic DNNs
# ===========================
class ActorNetwork(object):
    """ 
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -2 and 2
    """
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
        
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.mul(self.network_params[i], self.tau) + \
                tf.mul(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
        
        # Combine the gradients here 
        self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self): 
        inputs = tf.placeholder("float", [None, self.s_dim])
        # actor network u(s)
        # weight for policy output layer
        self.s_W_fc1 =  weight_variable([self.s_dim, 64])
        self.s_b_fc1 = bias_variable([64])
        self.s_W_fc2 =  weight_variable([64, 20])
        self.s_b_fc2 = bias_variable([20])
        a_h1_layer = tf.nn.sigmoid(tf.matmul(inputs, self.s_W_fc1) + self.s_b_fc1)
        a_h2_layer = tf.nn.sigmoid(tf.matmul(a_h1_layer, self.s_W_fc2) + self.s_b_fc2)

        # weight for value output layer
        self.W_fc3 = weight_variable([20, self.a_dim])
        self.b_fc3 = bias_variable([self.a_dim])

        out = tf.nn.tanh(tf.matmul(a_h2_layer, self.W_fc3)+ self.b_fc3)
        scaled_out = tf.mul(out, self.action_bound) # Scale output to -action_bound to action_bound
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def get_vars(self):
        return [self.s_W_fc1, self.s_b_fc1,
                    self.s_W_fc2, self.s_b_fc2,
                    self.W_fc3, self.b_fc3]

class CriticNetwork(object):
    """ 
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()
        
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.mul(self.network_params[i], self.tau) + tf.mul(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]
    
        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss =tf.reduce_mean(tf.square(self.predicted_q_value - self.out))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch 
        # w.r.t. that action (i.e., sum of dy/dx over all ys). We then divide
        # through by the minibatch size to scale the gradients down correctly.
        self.action_grads = tf.div(tf.gradients(self.out, self.action), tf.constant(BATCH_SIZE, dtype=tf.float32))

    def create_critic_network(self):
        inputs = tf.placeholder("float", [None, self.s_dim])
        action = tf.placeholder("float", [None, self.a_dim])
        
        self.s_W_fc1 =  weight_variable([self.s_dim, 64])
        self.s_b_fc1 = bias_variable([64])

        self.s_W_fc2 =  weight_variable([64, 20])
        self.s_b_fc2 = bias_variable([20])
        self.a_W_fc2 =  weight_variable([self.a_dim, 20])
        self.a_b_fc2 = bias_variable([20])

        self.W_critic_Q = weight_variable([20, 1])
        self.b_critic_Q = bias_variable([1])

        c_h1_layer = tf.nn.sigmoid(tf.matmul(inputs, self.s_W_fc1) + self.s_b_fc1)
        c_h2_layer = tf.nn.sigmoid(tf.matmul(c_h1_layer, self.s_W_fc2) + self.s_b_fc2 + tf.matmul(action, self.a_W_fc2) + self.a_b_fc2)
        
        out = tf.matmul(c_h2_layer, self.W_critic_Q) + self.b_critic_Q
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions): 
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_vars(self):
        return [self.s_W_fc1, self.s_b_fc1,
                    self.s_W_fc2, self.s_b_fc2,
                    self.a_W_fc2, self.a_b_fc2,
                    self.W_critic_Q, self.b_critic_Q]

class SimulatorNetwork(object):
    """ 
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """
    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.sess = sess
        self.si_dim = state_dim
        self.ai_dim = action_dim
        self.so_dim = state_dim
        self.learning_rate = learning_rate
        self.replay_buffer = deque()

        self.create_network()
        self.create_training_method()

        self.time_t = 0
        self.train_time = 1
        self.loss = 0
        self.time_step = 0

    def create_network(self):
            self.W1 =weight_variable([self.si_dim, 64])
            self.b1 = bias_variable([64])
            self.W2 =weight_variable([self.ai_dim, 2])
            self.b2 = bias_variable([2])

            self.W3 = weight_variable([66, self.so_dim])
            self.b3 =  bias_variable([self.so_dim])

            if DEBUG_MODE:
                    print "All weights and bias:"
                    print "W1: ", W1
                    print "b1: ", b1
                    print "W2: ", W2
                    print "b2: ", b2
                    print "W3:", W3
                    print "b3: ", b3
                    time.sleep(0.5)

            #input layer
            self.state_input = tf.placeholder("float", [None, self.si_dim])
            self.action_input = tf.placeholder("float", [None, self.ai_dim])
            
            # hidden layers
            h1_s_layer = tf.nn.relu(tf.matmul(self.state_input, self.W1) + self.b1)
            h1_a_layer = tf.nn.relu(tf.matmul(self.action_input, self.W2) + self.b2)
            h1_layer_concat = tf.concat(1, [h1_s_layer, h1_a_layer])

            self.state_output = tf.nn.relu(tf.matmul(h1_layer_concat, self.W3) + self.b3)

    def create_training_method(self):
                self.rt_nextstate = tf.placeholder("float", [None, self.si_dim])
                self.cost = tf.reduce_mean(tf.square(self.rt_nextstate - self.state_output))
                self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, state, action, next_state):
                self.time_t += 1
                
                self.replay_buffer.append((state, action, next_state))

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
                next_state_batch = [data[2] for data in minibatch]

                if DEBUG_MODE:
                        print "Training. Please wait." 
                        print "Fetch minibatch[0] to check: ---------------"
                        print "state_batch: ", state_batch[0][17:19]
                        # print "action_batch: ", action_batch[0]
                        time.sleep(0.5)

                # Step 2: calculate y
                self.optimizer.run(feed_dict={
                            self.rt_nextstate: next_state_batch,
                            self.action_input: action_batch,
                            self.state_input: state_batch
                    })

                if DEBUG_MODE:
                        print "this state", state_batch[0]
                        print "next state: ", next_state_batch[0]
                        # print "loss: ", loss/BATCH_SIZE
                        time.sleep(0.5)

    def predict_next(self, state, action):
                
                state_output = self.self.state_output.eval(feed_dict={
                            self.state_input: state,
                            self.action_input: action,
                            })[0]
                if DEBUG_MODE:
                        print "predict", state_output
                        time.sleep(0.5)
                return state_output


class DQN_CT():
        #DQN Continuous output for Agent
        def __init__(self, env):
                
                # ''env'' for car agent should be virtual sensor playground
                # env.observation should be 360 degree vec + distance + angle-diff
                # init experience replay
                self.replay_buffer = deque()
                # init some parameters
                self.time_step = 0
                self.state_dim = 362    # 360 vec + distance + angle diff
                self.action_dim = 2      # speed + steer
                self.speed_range = 5
                self.steer_range = 22.5
                
                self.session = tf.InteractiveSession()
                self.create_Q_network()
                self.session.run(tf.initialize_all_variables())

                self.all_saver = tf.train.Saver(tf.trainable_variables()[0:])
                # self.all_saver.restore(self.session, "./checkpoint/Use_acff/checkpoint-52")

                self.actor.update_target_network()
                self.critic.update_target_network()

                self.time_t = 0
                self.train_time = 1
                self.loss = 0
                self.simu_loss = 0
                self.epsilon = INITIAL_EPSILON

        def create_Q_network(self):
                self.actor = ActorNetwork(self.session, self.state_dim, self.action_dim, [self.speed_range, self.steer_range], \
                        ACTOR_LEARNING_RATE, TAU)

                self.critic = CriticNetwork(self.session, self.state_dim, self.action_dim, \
                        CRITIC_LEARNING_RATE, TAU, self.actor.get_num_trainable_vars())

                self.simulator =  SimulatorNetwork(self.session, self.state_dim, self.action_dim, \
                        CRITIC_LEARNING_RATE)


        def perceive(self, state, action, reward, next_state, done):
                self.time_t += 1
                
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
                y_batch = list()
                # print "a_b", len(action_batch[0]), len(action_batch[1])
                # print "here", len(next_state_batch[0]), len(next_state_batch[1])
                # print "here this", len(state_batch[0]), len(state_batch[1])
                # print next_state_batch[0], next_state_batch[1]
                
                target_q = self.critic.predict_target(next_state_batch, self.actor.predict_target(next_state_batch))

                for i in range(BATCH_SIZE):
                    done = minibatch[i][4]
                    if done:
                        y_batch.append(reward_batch[i])
                    else:
                        y_batch.append(reward_batch[i] + GAMMA * target_q[i])
                # print "y", y_batch

                # Update the critic given the targets
                predicted_q_value, _ = self.critic.train(state_batch, action_batch, np.reshape(y_batch, (BATCH_SIZE, 1)))
            
                
                a_outs = self.actor.predict(state_batch)                
                grads = self.critic.action_gradients(state_batch, a_outs)
                self.actor.train(state_batch, grads[0])

                # Update target networks
                self.actor.update_target_network()
                self.critic.update_target_network()

                
                self.loss += self.critic.sess.run([self.critic.loss], feed_dict={
                    self.critic.inputs: state_batch,
                    self.critic.action: action_batch,
                    self.critic.predicted_q_value: predicted_q_value
                })[0]

                self.simu_loss += self.simulator.sess.run([self.simulator.cost], feed_dict={
                    self.simulator.state_input: state_batch,
                    self.simulator.action_input: action_batch,
                    self.simulator.rt_nextstate: next_state_batch
                })[0]

                if DEBUG_MODE:
                        print "calculate y..." 
                        #print "Q_value_batch: ", Q_value_batch[0]
                        #print "y value: ", y_batch[0]
                        print "loss: ", self.loss
                        print "simu_loss", self.simu_loss
                        time.sleep(0.05)

        def egreedy_action(self, state):

                # a_explore = self.actor.predict(np.reshape(state+(random.random()*5), (-1, self.actor.s_dim)))
                a_deploy = self.actor.predict(np.reshape(state, (-1, self.actor.s_dim)))
                Q_value = self.critic.predict(np.reshape(state, (-1, self.critic.s_dim)), np.reshape(a_deploy, (-1, self.critic.a_dim)))


                if random.random() <= self.epsilon:
                        self.epsilon -= (self.epsilon - FINAL_EPSILON)/10000
                        ## Note!!!!
                        ## Initially, self.epsilon -= (INITIAL_EPSILON - FIANAL_EPSILON)/10000
                        ## But actually, result is after 10000/300 ~= 33 episode, exploration rate is 0.01
                        action = [0., 0.]
                        action = [random.random()*self.speed_range, random.random()*self.steer_range]
                
                else:
                        self.epsilon -= (self.epsilon - FINAL_EPSILON)/10000
                        action = [0., 0.]
                        action = a_deploy[0]

                if DEBUG_MODE:
                        print "Q_value:", Q_value
                        print "agent speed", action[0]
                        print "agent steer", action[1]
                        time.sleep(0.05)

                return action[0], action[1], Q_value

        def reset_loss(self):
                self.loss = 0
                self.simu_loss = 0
                return 1

        def get_loss(self):
                return self.loss

        def get_simu_loss(self):
                return self.simu_loss

        def get_vars(self):
                return self.actor.get_vars(), self.critic.get_vars()
