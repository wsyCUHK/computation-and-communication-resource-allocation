"""
Reinforcement Learning (A3C) using Pytroch + multiprocessing.
Testing
"""

import torch
import torch.nn as nn
from utils import v_wrap, set_init, push_and_pull, record
import torch.nn.functional as F
import torch.multiprocessing as mp
from shared_adam import SharedAdam
import gym
import os
import numpy as np
import random
import argparse


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--history_length', type=int, default=3, metavar='N',
                    help='the state length')
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 10000
MAX_EP_STEP = 201

#env = gym.make('Pendulum-v0')
#num_of_user=3



computation_upper_bound=2
bandwith_upper_bound=2
P=1000
N0=10**(-10)
candidate_bitrate=np.array([20,10,5,1])
num_of_bitrate=candidate_bitrate.shape[0]
N_S = 2+num_of_bitrate
history_length=args.history_length
N_S=history_length*N_S

#N_A=num_of_bitrate*(2**num_of_bitrate)
N_A=num_of_bitrate+num_of_bitrate*(2)
Computation_Cost=np.array([0,1,2,6,0])
a=0.01
b=0.05
c=[0]
loss2_penalty=0.5
beta=2

N=30
distv = 10+np.random.random_sample((N,1)) * 60


bandwidth=2*10**6

phone_arrival_rate=0.2
phone_departure_rate=0.2
phone_self_transition=0.8
TV_arrival_rate=0.05
TV_departure_rate=0.05
TV_self_transition=0.8
laptop_arrival_rate=0.1
laptop_departure_rate=0.1
laptop_self_transition=0.8
a_rate=[phone_arrival_rate,TV_arrival_rate,laptop_arrival_rate]
d_rate=[phone_departure_rate,TV_departure_rate,laptop_departure_rate]
s_tran=[phone_self_transition,TV_self_transition,laptop_self_transition]
T=2
kappa1=1
kappa2=1000
loss1_weight=1/3
loss2_weight=3/18
loss3_weight=14/57.12

def qoe_loss(cached_bitrates,request):
    loss=[]
    for i in range(request.shape[0]):
        small=1000
        for j in range(cached_bitrates.shape[0]):
            if cached_bitrates[j]>0.5:
                if (candidate_bitrate[request[i]]-candidate_bitrate[j])**2< small:
                    small=(candidate_bitrate[request[i]]-candidate_bitrate[j])**2
        loss.append(small)
    return np.sum(loss)

def function_mu(encoding_bitrate,transcoding_bitrate):
    transcoding_cpu_cycle=0
    for j in range(transcoding_bitrate.shape[0]):
        if transcoding_bitrate[j]>0.5:
            transcoding_cpu_cycle+=abs(candidate_bitrate[encoding_bitrate]-candidate_bitrate[j])
    return transcoding_cpu_cycle


def random_actor(num_of_bitrate):
    a=[random.randint(0,num_of_bitrate-1)]
    for i in range(num_of_bitrate):
        a.append(random.randint(0,1))
    return np.array(a)

def user_transit(users,state):
    new_users=[]
    for i in range(users.shape[0]):
        if random.random()>d_rate[users[i,1]]:
            new_u=users[i,:]
            if random.random()<s_tran[users[i,1]]:
                if state[2+new_u[0]]>0.5 or (state[2+min(new_u[0]+1,num_of_bitrate-1)]>0.5 and state[2+max(new_u[0]-1,0)]>0.5) or (state[2+min(new_u[0]+1,num_of_bitrate-1)]<0.5 and state[2+max(new_u[0]-1,0)]<0.5):
                    if random.random()>0.5:
                        new_u[0]=min(new_u[0]+1,num_of_bitrate-1)
                    else:
                        new_u[0]=max(new_u[0]-1,0)
                elif state[2+min(new_u[0]+1,num_of_bitrate-1)]>0.5:
                    new_u[0]= min(new_u[0]+1,num_of_bitrate-1)
                else:
                    new_u[0]=min(new_u[0]+1,num_of_bitrate-1)
            new_users.append(new_u)
            #New Arrival
    for i in range(3):
        if random.random()<a_rate[i]:
            new_users.append([random.randint(0,num_of_bitrate-1),i])
    return new_users

def env(a,s,users):
    ####################Map action in caterogrial##########################################
    caterogrial_action=np.round(a)
    caterogrial_action=caterogrial_action.astype('int')
    state=np.zeros((2+num_of_bitrate,))
    state[0]=s[0]
    state[1]=s[args.history_length]
    state[2:]=s[2*args.history_length:2*args.history_length+num_of_bitrate]
    ####################Model User Arrival, Departure, and Request Varying#################
    
    new_users=user_transit(users,state)
    new_users=np.array(new_users)
    encoding_bitrate=caterogrial_action[0]
    transcoding_bitrate=caterogrial_action[1:]
    for i in range(encoding_bitrate):
        transcoding_bitrate[i]=0
    transcoding_bitrate[encoding_bitrate]=1
    if new_users.shape[0]==0:
        loss1=0
    else:
        loss1=qoe_loss(transcoding_bitrate,new_users[:,0])
    loss2=candidate_bitrate[encoding_bitrate]*kappa1*T
    ##############Convex Optimization#############################################
    para_a=bandwidth*N0*1000/(state[0]*1000)
    para_b=candidate_bitrate[int(np.round(state[1]))]*(10**6)*T/bandwidth
    para_k3=function_mu(int(np.round(state[1])),transcoding_bitrate)
    value2=[]
    smallest=10**20
    index=0
    for xg in range(5,95):
        x=T/100*xg
        if (para_a*x*(np.power(2,para_b/x-1))+kappa2*(np.power(para_k3,3)/(np.power(T-x,2))))/(10**5)<smallest:
            index=x
            smallest=(para_a*x*(np.power(2,para_b/x-1))+kappa2*(np.power(para_k3,3)/(np.power(T-x,2))))/(10**5)
    loss3=smallest
    ##############Wireless Channel Model##########################################
    F_c = 915*1e6  # carrier bandwidth
    A_d = 79.43  # antenna gain
    degree = 2  # path loss value
    light = 3*1e8  # speed of light
    channel_u = np.zeros((50,N))
    channel_d = np.zeros((50,N))
    for j in range(N):
        for time_slot in range(50):
            channel_d[time_slot,j] = np.random.exponential() * A_d * (light / (4 * 3.141592653589793 * F_c * distv[j]))**degree
    state_2=np.zeros((num_of_bitrate*args.history_length,))
    for i in range(new_users.shape[0]):
        state_2[new_users[i,0]]=+1
    state_2[num_of_bitrate:]=s[2*args.history_length:N_S-num_of_bitrate]
    state_0=np.zeros((args.history_length,))
    state_0[0]=channel_d[0,0]
    state_0[1:]=s[:args.history_length-1]
    state_1=np.zeros((args.history_length,))
    state_1[0]=encoding_bitrate
    state_1[1:]=s[args.history_length:2*args.history_length-1]
    return -(loss1_weight*loss1+loss2_weight*loss2+loss3_weight*loss3),np.concatenate((np.concatenate((state_0,state_1),axis=0),state_2),axis=0),new_users,loss1_weight*loss1,loss2_weight*loss2,loss3_weight*loss3



class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        #self.pi1_1 = nn.Linear(2, 64)
        #self.pi1_2=nn.Linear(s_dim-2, 64)
        self.pi1=nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        #self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        #set_init([self.pi1, self.pi2, self.v2])
        self.distribution = torch.distributions.Categorical

        torch.nn.init.kaiming_normal_(self.pi1.weight)
        torch.nn.init.kaiming_normal_(self.pi2.weight)
        torch.nn.init.kaiming_normal_(self.v2.weight)

    def forward(self, x):
        #print(x.shape)
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        #v1 = torch.tanh(self.v1(x))
        values = self.v2(pi1)
        return logits, values

    def choose_action(self, s):
        self.eval()
        #print('choosing')
        logits, _ = self.forward(s)
        #print(logits.shape)
        #print(logits[0][:num_of_bitrate].shape)
        #print(logits[0,:num_of_bitrate].shape)
        prob=[]
        prob.append(F.softmax(logits[0,:num_of_bitrate], dim=-1).data)
        for i in range(num_of_bitrate):
            prob.append(F.softmax(logits[0,num_of_bitrate+i*2:num_of_bitrate+(i+1)*2], dim=-1).data)
        m_output=np.zeros((num_of_bitrate+1,))
          		#print(prob)
        m_output[0]=(self.distribution(prob[0]).sample().numpy())
          		#print(m_output[-1])
        for i in range(num_of_bitrate):
          			#print(self.distribution(prob[1+i]).sample().numpy())
          			m_output[1+i]=(self.distribution(prob[1+i]).sample().numpy())
          		#print(m_output)
        return m_output

    def loss_func(self, s, a, v_t):
        self.train()
        #print('training')
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        #print(logits.shape)
        #print(logits)
        #print(logits[:,:num_of_bitrate].shape)
        prob=[]
        m1=self.distribution(F.softmax(logits[:,:num_of_bitrate], dim=-1).data)
        #print(a.shape)
        m=m1.log_prob(a[:,0])
        for i in range(num_of_bitrate):
        	m1=self.distribution(F.softmax(logits[:,num_of_bitrate+i*2:num_of_bitrate+(i+1)*2], dim=-1).data)
        	m=m*m1.log_prob(a[:,1+i])

        #probs = F.softmax(logits, dim=1)
        #m = self.distribution(probs)
        exp_v = m * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

def env_reset():
    # Generate the channel gain for next time slot
    F_c = 915*1e6  # carrier bandwidth
    A_d = 79.43  # antenna gain
    degree = 2  # path loss value
    light = 3*1e8  # speed of light
    channel_u = np.zeros((50,N))
    channel_d = np.zeros((50,N))


    for j in range(N):
        for time_slot in range(50):
            channel_d[time_slot,j] = np.random.exponential() * A_d * (light / (4 * 3.141592653589793 * F_c * distv[j]))**degree
    s=np.array([channel_d[0,0],random.randint(0,3),random.randint(0,1),random.randint(0,1),random.randint(0,1),random.randint(0,1)])
    s=np.zeros((N_S,))
    s[0]=channel_d[0,0]
    s[args.history_length]=random.randint(0,3)
    if random.random()<0.01:
        u=np.expand_dims(np.array([random.randint(0,3),random.randint(0,2)]), axis=0)
    elif random.random()<0.1:
        u=np.array([[random.randint(0,3),random.randint(0,2)],[random.randint(0,3),random.randint(0,2)]])
    else:
        u=np.array([[random.randint(0,3),random.randint(0,2)],[random.randint(0,3),random.randint(0,2)],[random.randint(0,3),random.randint(0,2)]])
    for i in range(history_length):
        action=np.array([random.randint(0,3),random.randint(0,1),random.randint(0,1),random.randint(0,1),random.randint(0,1)])
        _, s_, u_ ,_,_,_= env(action,s.squeeze(),u)
        s=s_
        u=u_
    return action,s,u

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net(N_S, N_A)           # local network
        self.env = env

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            ################################
            #       Initial State          #
            ################################
            a, s,u=env_reset()
            s=torch.tensor(s).reshape((1,N_S)).float()
            #########################
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            for t in range(MAX_EP_STEP):
                #print('Time: '+str(t))
                # if self.name == 'w00':
                #     self.env.render()
                #print('We are choosing action!')
                #print(s.shape)
                #print(N_A)
                #print(s.shape)
                a = self.lnet.choose_action(s)
                #print(a)
                #print(s.shape)
                r, s_, u_,_,_,_= self.env(a,s.squeeze(),u)
                s_=torch.tensor(s_).float()
                r=np.expand_dims(np.expand_dims(r, 0), 0)
                s_=s_.reshape((1,N_S)).float()
                #print('NExt State')
                #print(s_.shape)
                #if done: r = -1
                ep_r += r
                #a=
                #print(a.shape)
                #print(a.squeeze().shape)
                #print(s.shape)
                #print(s.squeeze().numpy().shape)
                #print(r.shape)
                #print(a)
                #print(r)
                buffer_a.append(np.array(a))
                buffer_s.append(s.squeeze().numpy())
                buffer_r.append(r.squeeze())
                done=False
                if t == MAX_EP_STEP - 1:
                    done = True
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                u=u_
                total_step += 1
        self.res_queue.put(None)


if __name__ == "__main__":
    total_res=[]
    epi=10
    for experiments in range(epi):
        gnet = Net(N_S, N_A)        # global network
        gnet.share_memory()         # share the global parameters in multiprocessing
        opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
        global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

        # parallel training
        workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
        #workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(1)]
        [w.start() for w in workers]
        res = []                    # record episode reward to plot
        while True:
            r = res_queue.get()
            if r is not None:
                res.append(r)
            else:
                break
        [w.join() for w in workers]
        total_res.append(np.mean(res))
    import matplotlib.pyplot as plt
    plt.plot(total_res)
    plt.ylabel('Average Reward')
    plt.xlabel('Sample Index')
    plt.show()
    #total_res.append(res)

    # import pickle
    # with open('result_l'+str(history_length)+'.pickle', 'wb') as f:
    #     pickle.dump(total_res, f)

    # import matplotlib.pyplot as plt
    # outputarray=np.zeros((epi,MAX_EP))
    # for i in range(epi):
    #     for j in range(MAX_EP):
    #         outputarray[i,j]=total_res[i][j]
    # total_res=outputarray
    # mean_plot=np.mean(total_res,0)
    # from scipy.stats import norm

    # upper=np.zeros((total_res.shape[1],))
    # lower=np.zeros((total_res.shape[1],))
    # for experiments in range(total_res.shape[1]):
    #     ci = norm(*norm.fit(total_res[:,experiments])).interval(0.90)
    #     upper[experiments]=ci[0]
    #     lower[experiments]=ci[1]

    

    # #plt.plot(mean_plot)
    # plt.plot(np.arange(1,total_res.shape[1]+1), mean_plot, color='purple', lw=0.5, ls='-', marker='o', ms=4)
    # plt.fill_between(np.arange(1,total_res.shape[1]+1), upper, lower, color=(229/256, 204/256, 249/256), alpha=0.9)
    # plt.ylabel('Moving average ep reward')
    # plt.xlabel('Step')
    # plt.show()
