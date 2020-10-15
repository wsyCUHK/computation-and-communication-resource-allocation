import os
import random
import sys
import time

#import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


beta=2
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

if not os.path.exists('epoch_100'):
    os.makedirs('epoch_100')
my_path='./epoch_100/'

#bandwith_upper_bound=2
num_of_user=3

for bandwith_upper_bound in range(1,6):

    num_of_act=(5**num_of_user)*(num_of_user*num_of_user)  
    computation_upper_bound=2
    
    P=1000
    N0=10**(-10)
    candidate_bitrate=np.array([20,10,5,1,0])
    Computation_Cost=np.array([0,1,2,6,0])
    a=0.01
    b=0.05
    c=[0]
    loss2_penalty=0.5
    N = 30
    distv = 10+np.random.random_sample((N,1)) * 60

    if num_of_user==1:
    	one_hot_vector=np.array([[1]])
    elif num_of_user==2:
    	one_hot_vector=np.array([[1,0],[0,1]])
    elif num_of_user==3:
    	one_hot_vector=np.array([[1,0,0],[0,1,0],[0,0,1]])
    elif num_of_user==4:
    	one_hot_vector=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    else:
    	one_hot_vector=np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
    def env(action,state,t):
       
        action=action.cpu().numpy()
        state=state.cpu().numpy()

        
        F_c = 915*1e6  # carrier bandwidth
        A_d = 79.43  # antenna gain
        degree = 2  # path loss value
        light = 3*1e8  # speed of light

        ch_idx = 100000

        channel_u = np.zeros((1,N))
        channel_d = np.zeros((1,N))

        
        for j in range(N):
            channel_d[0,j] = np.random.exponential() * A_d * (light / (4 * 3.141592653589793 * F_c * distv[j]))**degree
        #channel_d=torch.tensor(channel_d,device=device)

        #print(action)
        #action=action_index
        state=state.squeeze().squeeze()
        demand=[]
        for user in range(num_of_user):
            demand.append(action%5)
            action=int(action/5)
        #t=0
        #one_hot_vector=np.array([[1,0,0],[0,1,0],[0,0,1]])
        cp_user=action%num_of_user
        #cp_user=0
        #bd_user=0
        action=int(action/num_of_user)
        bd_user=action%num_of_user
        downlink_rate=bandwith_upper_bound*np.log2(1+channel_d[0,bd_user]*P/(N0*np.dot(1000000,bandwith_upper_bound)))
        action=int(action/num_of_user)
        
        #d2=action%5
        #action=int(action/5)
        #d3=action%5
        #demand=[action]
        bitrate_torch=np.array([candidate_bitrate[d] for d in demand])
        residual_computation=state[num_of_user:2*num_of_user]+bitrate_torch
        residual_computation=residual_computation-one_hot_vector[cp_user]*computation_upper_bound
        residual_computation[residual_computation<0]=0
        not_avaiable_download=state[3*num_of_user:4*num_of_user]+bitrate_torch*beta
        residual_communication=state[2*num_of_user:3*num_of_user]#+bitrate_torch*beta
        for user in range(num_of_user):
            if residual_computation[user]<0.01:
                residual_communication[user]=residual_communication[user]+not_avaiable_download[user]
                not_avaiable_download[user]=0

        residual_buffer=state[0:num_of_user]
        effect_buffer=residual_buffer
        loss2=0
        for user in range(num_of_user):
            effect_buffer[user]=2*int(effect_buffer[user]/2)
            if effect_buffer[user]>1:
                residual_buffer[user]-=1
            else:
                loss2+=1

        if residual_communication[bd_user]-downlink_rate>0:
            residual_communication[bd_user]=residual_communication[bd_user]-downlink_rate
            residual_buffer[bd_user]+=downlink_rate
        else:
            residual_buffer[bd_user]+=residual_communication[bd_user]
            residual_communication[bd_user]=0

        last_rate=state[4*num_of_user:5*num_of_user]
        loss1=0
        for user in range(num_of_user):
            if candidate_bitrate[demand[user]]>0:
                loss1+=np.dot(a,((last_rate[user]-candidate_bitrate[demand[user]])*(last_rate[user]-candidate_bitrate[demand[user]])))+np.dot(b,(last_rate[user]-candidate_bitrate[demand[user]]))+c
                last_rate[user]=candidate_bitrate[demand[user]]
            else:
            	last_rate[user]=state[4*num_of_user+user]
            	loss1=[0]
        # if candidate_bitrate[d2]>0:
        #     last_rate[0]=candidate_bitrate[d2]
        # else:
        #     last_rate[0]=state[13]
        # if candidate_bitrate[d3]>0:
        #     last_rate[0]=candidate_bitrate[d3]
        # else:
        #     last_rate[0]=state[14]
        qoe=np.sum([np.log2(1+candidate_bitrate[d]) for d in demand])-loss1-loss2_penalty*loss2

        #print(qoe.shape)
        #print(residual_buffer.shape)
        #print(residual_computation.shape)
        #print(residual_communication.shape)
        #print(not_avaiable_download.shape)
        #print(last_rate.shape)
        return np.array(qoe),bitrate_torch,loss1,loss2,np.concatenate((residual_buffer, residual_computation, residual_communication, not_avaiable_download, last_rate))


    class NeuralNetwork(nn.Module):

        def __init__(self):
            super(NeuralNetwork, self).__init__()

            self.number_of_actions = num_of_act
            self.gamma = 0.99
            self.final_epsilon = 0.0001
            self.initial_epsilon = 0.1
            self.number_of_iterations = 100000
            self.replay_memory_size = 10000
            self.minibatch_size = 32

            #self.conv1 = nn.Conv2d(4, 32, 8, 4)
            self.conv1 =nn.Conv1d(1,1,3,padding=2)
            self.relu1 = nn.ReLU(inplace=True)
            self.maxpool1d=nn.MaxPool1d(2, stride=1)
            #self.flatten1=torch.flatten()
            #self.conv2 = nn.Conv2d(32, 64, 4, 2)
            #self.relu2 = nn.ReLU(inplace=True)
            #self.conv3 = nn.Conv2d(64, 64, 3, 1)
            #self.relu3 = nn.ReLU(inplace=True)
            self.fc4 = nn.Linear(num_of_user*5+1, 64)
            self.relu4 = nn.ReLU(inplace=True)
            self.fc5 = nn.Linear(64, self.number_of_actions)

        def forward(self, x):
            #print(x.shape)
            out = self.conv1(x)
            out = self.relu1(out)
            out = self.maxpool1d(out)
            out = self.fc4(out)
            out = self.relu4(out)
            out = self.fc5(out)

            return out


    def init_weights(m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            torch.nn.init.uniform(m.weight, -0.01, 0.01)
            m.bias.data.fill_(0.01)

    

    model = NeuralNetwork()

    if USE_CUDA:  # put on GPU if CUDA is available
        model = model.cuda()

    model.apply(init_weights)
    start = time.time()

    # define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

    # instantiate game
    #game_state = GameState()

    residual_buffer=torch.zeros((num_of_user,1))
    residual_computation=torch.zeros((num_of_user,1))
    residual_communication=torch.zeros((num_of_user,1))
    not_avaiable_download=torch.zeros((num_of_user,1))
    last_rate=torch.zeros((num_of_user,1))
    # initialize replay memory
    replay_memory = []

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    #image_data, reward, terminal = game_state.frame_step(action)
    #image_data = resize_and_bgr2gray(image_data)
    #image_data = image_to_tensor(image_data)
    state = torch.cat((residual_buffer, residual_computation, residual_communication,not_avaiable_download,last_rate)).reshape((1,5*num_of_user)).unsqueeze(0)

    print(state.shape)

    # initialize epsilon value
    epsilon = model.initial_epsilon
    iteration = 0

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    # main infinite loop
    while iteration < model.number_of_iterations:
        # get output from the neural network
        output = model(state.cuda())[0]

        # initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        if random_action:
            print("Performed random action!")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()

        action[action_index] = 1

        # get next state and reward
        #image_data_1, reward, terminal = game_state.frame_step(action)
        rd,metric1,metric2,metric3,state_1=env(action_index,state,iteration)
        #image_data_1 = resize_and_bgr2gray(image_data_1)
        #image_data_1 = image_to_tensor(image_data_1)
        #state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.zeros([model.number_of_actions], dtype=torch.float32)
        reward[action_index] = torch.from_numpy(rd).float()
        reward=reward.unsqueeze(0).unsqueeze(0)
        #print(reward.shape)
        state_1=torch.from_numpy(state_1).unsqueeze(0).unsqueeze(0).float()
        # save transition to replay memory
        replay_memory.append((state.cpu(), action.cpu(), reward, state_1, 0))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        # unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # get output for the next state
        output_1_batch = model(state_1_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        # extract Q-value
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        #print(q_value.shape)
        #print(y_batch.shape)
        # calculate loss
        loss = criterion(q_value, y_batch)

        # do backward pass
        loss.backward()
        optimizer.step()

        # set state to be state_1
        state = state_1
        iteration += 1

        if iteration % 25000 == 0:
            torch.save(model, "pretrained_model/current_"+str(num_of_user)+"user_model_" + str(iteration) + ".pth")

        if iteration % 500 == 0:
            print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
              action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
              np.max(output.cpu().detach().numpy()))




    for test_epoch in range(100):
        residual_buffer=torch.zeros((num_of_user,1))
        residual_computation=torch.zeros((num_of_user,1))
        residual_communication=torch.zeros((num_of_user,1))
        not_avaiable_download=torch.zeros((num_of_user,1))
        last_rate=torch.zeros((num_of_user,1))
        # initialize replay memory
        #replay_memory = []

        # initial action is do nothing
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        action[0] = 1
        #image_data, reward, terminal = game_state.frame_step(action)
        #image_data = resize_and_bgr2gray(image_data)
        #image_data = image_to_tensor(image_data)
        state = torch.cat((residual_buffer, residual_computation, residual_communication,not_avaiable_download,last_rate)).reshape((1,5*num_of_user)).unsqueeze(0)

        print(state.shape)


        iteration = 0
        qoe_log=[]
        bitrate_log=[]
        loss1_log=[]
        loss2_log=[]
        # main infinite loop
        
        with torch.no_grad():
            while iteration < 100:
                # get output from the neural network
                state=state.cuda()
                output = model(state)[0]

                # initialize action
                action = torch.zeros([model.number_of_actions], dtype=torch.float32)
                if torch.cuda.is_available():  # put on GPU if CUDA is available
                    action = action.cuda()



                action_index=torch.argmax(output)

                if torch.cuda.is_available():  # put on GPU if CUDA is available
                    action_index = action_index.cuda()

                action[action_index] = 1

                # get next state and reward
                #image_data_1, reward, terminal = game_state.frame_step(action)
                qoe,bitrate,loss1,loss2,state_1=env(action_index,state,iteration)
                state_1=torch.from_numpy(state_1).unsqueeze(0).unsqueeze(0).float()
                #image_data_1 = resize_and_bgr2gray(image_data_1)
                #image_data_1 = image_to_tensor(image_data_1)
                #state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

                action = action.unsqueeze(0)
                #reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
                qoe_log.append(qoe)
                bitrate_log.append(bitrate)
                loss1_log.append(loss1)
                loss2_log.append(loss2)
                
                state = state_1
                iteration += 1

                

        filename=my_path+'dqn_epoch'+str(test_epoch)+'_'+str(bandwith_upper_bound)+'bandwidth_20200929.pickle'
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump([qoe_log,bitrate_log,loss1_log,loss2_log], f)