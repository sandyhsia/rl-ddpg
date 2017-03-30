import tensorflow as tf
import numpy as np
import random
import os
from collections import deque
from Virtual_Env_CTmulti import *
from DQN_CT_simu import *
import time as time

# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'ACFF'
EPISODE = 10000 # Episode limitation
STEP = 3000 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode
FC_CHECKPOINT_DIR = './checkpoint/Use_fc'
Conv_CHECKPOINT_DIR = './checkpoint/Use_conv'
LSTM_CHECKPOINT_DIR = './checkpoint/Use_lstm'
ACFF_CHECKPOINT_DIR = './checkpoint/Use_acff'
USE_FC_ONLY = 0 # state is a 362-dim vector
USE_CONV = 0 # state is a 20x20-dim vector
USE_LSTM = 0 # adding rnn to the DQN
USE_ACFF = 1
save_request = 0
restore_request = 0
want_test = 0
want_log = 0
testlog = './' + 'logFile'+ENV_NAME
alpha = 0.95
car_agent_num = 1
agent_array = list()
saver = list()

def main():
    start_time = time.time()
    saver = list()
    # initialize OpenAI Gym env and dqn agent
    env = Virtual_Env(ENV_NAME, 100, 100)
    for i in range(car_agent_num):
        print "restoring agent", i, "......"
        agent = DQN_CT(env)
        agent_array.append(agent)
        agent.all_saver.restore(agent.session, './checkpoint/Use_acff/A0.chkp-1')
        
    # s1 = tf.train.Saver(agent_array[0].get_vars())
    # s2 = tf.train.Saver()
        # saver[i].restore(agent_array[i].session, './checkpoint/Use_acff/checkpoint-103')
    print "restore done."
    
    # saver = tf.train.Saver()
    # saver = tf.train.import_meta_graph('./checkpoint/Use_acff/checkpoint-3495.meta')
    global_t = 0
    CHECKPOINT_DIR = choose_dir()



    if restore_request == 1:
            checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
            if checkpoint and checkpoint.model_checkpoint_path:
                    saver.restore(agent_array[0].session, './checkpoint/Use_acff/checkpoint-3495')
                    # saver.restore(agent_array[1].session, './checkpoint/Use_acff/checkpoint-103')
                    tokens = checkpoint.model_checkpoint_path.split("-")
                    # set global step
                    global_t = int(tokens[1])
                    print(">>> global step set: ", global_t)
                    print("checkpoint loaded:", checkpoint.model_checkpoint_path)
            else:
                    print("Could not find old checkpoint")

    if want_log:
            init_logfile(testlog, global_t)

    try:
            for episode in range(EPISODE):
                    # initialize task
                    print "episode", episode
                    state = env.reset(0, 0)
                    time.sleep(0.5)
                    # Train
                    for step in range(STEP):
                            
                            if step == 0:
                                    time_step = agent_array[0].time_step

                            action_array = list()
                            for i in range(car_agent_num):
                                    action_speed, action_steer, Q = agent_array[i].egreedy_action(state[i]) # e-greedy action for train
                                    action = [action_speed, action_steer]
                                    action_array.append(action)
                                    # print action
                            next_state,reward,done = env.step_multimode(env.car_center, env.angle, action_array)

                            for i in range(car_agent_num):
                                    # print i
                                    action = action_array.pop(0)
                                    agent_array[i].perceive(state[i], action, reward[i], next_state[i], done[i])
                                    agent_array[i].simulator.perceive(state[i], action, next_state[i])
                            state = next_state
                            for i in range(car_agent_num):
                                    if done[i] == True:
                                            print "simu_loss", agent_array[i].get_simu_loss()
                                            print_stats_log(episode, env.eat_up_time[i], step, time_step, agent_array[i].time_step, agent_array[i].get_loss())
                                            agent_array[i].reset_loss()
                                            env.reset(0, 0)

                            if step == STEP - 1:
                                    for i in range(car_agent_num):
                                            print "simu_loss", agent_array[i].get_simu_loss()
                                            print_stats_log(episode, env.eat_up_time[i], step, time_step, agent_array[i].time_step, agent_array[i].get_loss())
                                            agent_array[i].reset_loss()
                            # time.sleep(0.01)


                    # Test every 100 episodes
                    if episode % 100 == 0 and want_test:
                            print "---Test---"
                            total_reward = 0
                            for i in xrange(TEST):
                                    state = env.reset(0)
                                    time.sleep(0.5)
                                    for j in xrange(STEP):
                                            if step == 0:
                                                    time_step = agent.time_step

                                            action_speed, action_steer, Q = agent.egreedy_action(state) # e-greedy action for train
                                            action = (action_speed, action_steer)
                                            # print action
                                            next_state,reward,done = env.step(env.car_center, env.angle, action)
                                            if done == True:
                                                    break
                                            # time.sleep(0.01)
                            ave_reward = total_reward/TEST
                            print 'episode: ',episode,'Evaluation Average Reward:',ave_reward


    except KeyboardInterrupt:
            print "----You press Ctrl+C.----"
            if save_request == 1:
                    print "Now saving... wait."  
                    # write wall time
                    wall_t = time.time() - start_time
                    wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t+episode)
                    with open(wall_t_fname, 'w') as f:
                            f.write(time.ctime())
                            f.write(str(wall_t))
                    saver.save(agent.session, CHECKPOINT_DIR + '/' + 'checkpoint', global_step =global_t+episode)
                    print "Save done."
            else:
                    print "want save? [y/n]:"
                    input = raw_input()
                    if input == 'y':
                            print "Now saving... wait."  
                            # write wall time
                            wall_t = time.time() - start_time
                            wall_t_fname = CHECKPOINT_DIR + '/' + 'wall_t.' + str(global_t+episode)
                            with open(wall_t_fname, 'w') as f:
                                    f.write(time.ctime())
                                    f.write(str(wall_t))
                            
                            agent.all_saver.save(agent_array[0].session, CHECKPOINT_DIR + '/'+'A'+str(i)+'.chkp', global_step =global_t+episode)
                            # s2.save(agent_array[1].session, CHECKPOINT_DIR + '/'+'A'+str(i)+'.chkp', global_step =global_t+episode)
                            print "Save done."


def choose_dir():
        if USE_FC_ONLY == 1:
                CHECKPOINT_DIR = FC_CHECKPOINT_DIR
        elif USE_CONV == 1:
                CHECKPOINT_DIR = Conv_CHECKPOINT_DIR
        elif USE_LSTM == 1:
                CHECKPOINT_DIR = LSTM_CHECKPOINT_DIR
        elif USE_ACFF == 1:
                CHECKPOINT_DIR = ACFF_CHECKPOINT_DIR
        
        if not os.path.exists(CHECKPOINT_DIR):
                os.mkdir(CHECKPOINT_DIR)
        return CHECKPOINT_DIR

def print_stats_log(episode, eat_up_time, step, init_time_step, new_time_step, loss):
        print "Eat-up in epi ",episode,": ", eat_up_time
        print "Done in step: ",step
        if new_time_step == init_time_step:
                print "Haven't trained yet."
        else:
                print "Average train loss in", (new_time_step - init_time_step), "round:", loss/(new_time_step - init_time_step)
        print ""
        
        if want_log == 1:
                with open(testlog, 'a') as f:
                        f.write("espisode: "+str(episode)+'\n')
                        f.write("eat-up: " +str(eat_up_time)+'\n')
                        f.write("Done in step: "+str(step)+'\n')
                        if new_time_step != init_time_step:
                                f.write("Average train loss in"+str(new_time_step - init_time_step)+"round:"+str(loss/(new_time_step - init_time_step))+'\n')
                        f.write('\n')
        return

def init_logfile(filename, global_t):
        with open(testlog, 'a') as f:
                f.write("Training starts at "+time.ctime()+'\n')
                f.write("restore request is: "+str(restore_request)+"from "+str(global_t)+'\n')
                f.write("save request is: "+str(save_request)+'\n')
                f.write("Use method from: "+choose_dir()+'\n')
                f.write('\n')




if __name__ == '__main__':
        main()