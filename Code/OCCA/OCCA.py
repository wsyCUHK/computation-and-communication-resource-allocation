# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 20:19:20 2020

@author: sywang
"""


import numpy as np
import math
N = 30

distv = 10+np.random.random_sample((N,1)) * 60
F_c = 915*1e6  # carrier bandwidth
A_d = 79.43  # antenna gain
d = 2  # path loss value
light = 3*1e8  # speed of light

ch_idx = 100

channel_u = np.zeros((ch_idx, N))
channel_d = np.zeros((ch_idx, N))

for i  in range(ch_idx):
    for j in range(N):
        channel_d[i, j] = np.random.exponential() * A_d * (light / (4 * 3.141592653589793 * F_c * distv[j]))**d



video_length=[10,20,30,40,50,60,70]
probability=[0.2,0.3,0.1,0.1,0.2,0.1]

cdf=np.zeros((5,1))
for i in range(5):
    cdf[i]=sum(probability[0:i+1])


num_of_user=3;
user_video_length=np.zeros((num_of_user,1))
for i in range(num_of_user):
    temp1=np.random.random(1)
    for j in range(5):
        if temp1<cdf[j]:
            user_video_length[i]=video_length[j]
            break


Quality_reward_b=1
P0=1
Total_Bandwidth=100000

#Initial
residual_buffer=np.zeros((num_of_user,1))
new_buffer=np.zeros((num_of_user,1))
residual_computation=np.zeros((num_of_user,1))
residual_communication=np.zeros((num_of_user,1))
new_release_download=np.zeros((num_of_user,1))
downloading_rate=np.zeros((num_of_user,1))
last_rate=np.zeros((num_of_user,1))
current_dec_bit=np.zeros((num_of_user,1))
tool_vector1=[1,0,0]
tool_vector2=[0,1,0]
tool_vector3=[0,0,1]

tool_sum_vector=[1,1,1]

a=0.01
b=0.05
c=0
loss2_penalty=0.5
#x1=10
# cvx_begin
# variables x2(num_of_user) x3(num_of_user) y(num_of_user);
# maximize(log(x1)-a*x1^2-b*x1-inv_pos(y)'*(20-x1)*tool_vector1-inv_pos(y)'*residual_computation-inv_pos(x3)'*(20-x1)*tool_vector1-inv_pos(x3)'*residual_communication);
# subject to
#     tool_sum_vector*x2 <= 5
#     tool_sum_vector*x3 <= 5
#     y-inv_pos(x2.*log(1+inv_pos(x2))) >= 0
#     x2 >= 0
#     x3 >= 0
# cvx_end
computation_upper_bound=2
bandwith_upper_bound=2
beta=2 # 2 seconds
P=1000
N0=10**(-10)

Computation_Cost=np.array([0,1,2,6])
candidate_bitrate=np.array([20,10,5,1])
num_of_bit=4
one_hot_vector=np.array([[1,0,0],[0,1,0],[0,0,1]])
t=0
next_time_slot=0
user_id=1

#last_rate=0

downlink_log=[]
cp_log=[]
delay_log=[]
bitrate_log=[]
while next_time_slot<100:
    t=next_time_slot
    lamda=channel_d[t,0:num_of_user]*P/N0
    #print('It is time %i, the residual is (%.2f,%.2f,%.2f) and (%.2f,%.2f,%.2f)'%(t,residual_computation[0][0],residual_computation[1][0],residual_computation[2][0],residual_communication[0][0],residual_communication[1][0],residual_communication[2][0]))
    if np.min(residual_computation+residual_communication+new_release_download)<0.0001:
        try:
            user_id=np.argmin(residual_computation+residual_communication+new_release_download)[0]
        except:
            user_id=np.argmin(residual_computation+residual_communication+new_release_download)
        qoe=[]
        decision_computation_history=[]
        decision_bandwidth_history=[]
        deep_look1=[]
        deep_look2=[]
        deep_look3=[]
        for i in range(4):
            temp_residual_computation=residual_computation+np.reshape(one_hot_vector[user_id]*(Computation_Cost[i]),(num_of_user,1))
            #decision_computation=computation_upper_bound*(np.sqrt(temp_residual_computation)/sum(np.sqrt(temp_residual_computation)))
            if sum(np.sqrt(temp_residual_computation))<0.001:
                decision_computation=np.zeros((num_of_user,1))
            else:
                decision_computation=computation_upper_bound*(np.sqrt(temp_residual_computation)/sum(np.sqrt(temp_residual_computation)))
            temp_residual_communication=residual_communication+np.reshape(one_hot_vector[user_id]*candidate_bitrate[i]*beta,(num_of_user,1))

            weight=0
            for j in range(num_of_user):
                weight+=lamda[user_id]*np.sqrt(temp_residual_communication)[j]/(lamda[j]*np.sqrt(temp_residual_communication)[user_id])
            unit_bandwidth=(bandwith_upper_bound+(weight-num_of_user)/2)/weight
            decision_bandwidth=[]
            for j in range(num_of_user):
                decision_bandwidth.append(unit_bandwidth*lamda[user_id]*np.sqrt(temp_residual_communication)[j]/(lamda[j]*np.sqrt(temp_residual_communication)[user_id])-(lamda[user_id]*np.sqrt(temp_residual_communication)[j]/(lamda[j]*np.sqrt(temp_residual_communication)[user_id])-1)/2)
            downlink_rate=decision_bandwidth*np.log2(1+np.reshape(lamda,(num_of_user,1))/np.dot(1000000,np.array(decision_bandwidth)))
            downlink_rate[np.isnan(downlink_rate)]=0
            loss1=np.dot(a,((last_rate[user_id]-candidate_bitrate[i])*(last_rate[user_id]-candidate_bitrate[i])))+np.dot(b,(last_rate[user_id]-candidate_bitrate[i]))+c
            l1=temp_residual_communication/downlink_rate
            l1[np.isnan(l1)] = 0
            l2=temp_residual_computation/decision_computation
            l2[np.isnan(l2)] = 0
            loss2=l1+l2-residual_buffer
            
            qoe.append(np.log2(1+candidate_bitrate[i])-loss1-loss2_penalty*sum(loss2))
            deep_look1.append(np.log2(1+candidate_bitrate[i]))
            deep_look2.append(loss1)
            deep_look3.append(loss2_penalty*sum(loss2))
            decision_computation_history.append(decision_computation)
            decision_bandwidth_history.append(decision_bandwidth)

        index=np.argmax(qoe)
        current_dec_cp=decision_computation_history[index]
        current_dec_bd=decision_bandwidth_history[index]
        current_dec_bit[user_id]=candidate_bitrate[index]
        last_rate[user_id]=candidate_bitrate[index]
        residual_computation=residual_computation+np.reshape(one_hot_vector[user_id]*(Computation_Cost[index]),(num_of_user,1))
        new_release_download=new_release_download+np.reshape(one_hot_vector[user_id]*candidate_bitrate[index]*beta,(num_of_user,1))
        #downloading_rate[user_id]=candidate_bitrate[index]
        
        if np.min(residual_computation+residual_communication+new_release_download)<0.0001:
            next_time_slot=t
        else:
            residual_computation=residual_computation-current_dec_cp
            residual_computation[residual_computation<0]=0
            downlink_rate=np.array(current_dec_bd)*np.log2(1+np.reshape(lamda,(num_of_user,1))/np.dot(1000000,np.array(current_dec_bd)))
            downlink_rate[np.isnan(downlink_rate)]=0
            if sum(current_dec_cp)<0.01:
                for user in range(num_of_user):
                    if current_dec_bit[user]>19.9:
                        residual_communication[user]+=current_dec_bit[user]*beta
                        new_release_download[user]-=current_dec_bit[user]*beta
                #There is no new relesea
                old_rc=residual_communication
                residual_communication=residual_communication-downlink_rate
                #print(residual_communication)
                residual_communication[residual_communication<0]=0
                for user in range(num_of_user):
                    if old_rc[user]>0.01 and residual_communication[user]<0.01:
                        new_buffer[user]+=beta
            else:
                residual_computation=residual_computation-current_dec_cp
                residual_computation[residual_computation<0]=0
                downlink_rate=np.array(current_dec_bd)*np.log2(1+np.reshape(lamda,(num_of_user,1))/np.dot(1000000,np.array(current_dec_bd)))
                downlink_rate[np.isnan(downlink_rate)]=0
                
                time_cp=residual_computation/current_dec_cp
                time_cp[np.isnan(time_cp)]=0
                download_time_for_new_release=1-time_cp
                
                    
                download_time_for_new_release[np.isnan(download_time_for_new_release)]=0
                download_time_for_new_release[download_time_for_new_release<0]=0
                #print(residual_communication)
                residual_communication=residual_communication-(1-download_time_for_new_release)*downlink_rate
                #print(residual_communication)
                residual_communication[residual_communication<0]=0
                #print(residual_communication)
                for user in range(num_of_user):
                    if residual_computation[user]<0.0001:
                        residual_communication=residual_communication+np.reshape(one_hot_vector[user]*new_release_download[user],(num_of_user,1))
                #print(residual_communication)
                old_rc=residual_communication
                residual_communication=residual_communication-(download_time_for_new_release)*downlink_rate
                #print(residual_communication)
                residual_communication[residual_communication<0]=0
                for user in range(num_of_user):
                    if old_rc[user]>0.01 and residual_communication[user]<0.01:
                        new_buffer[user]+=beta
                #print(residual_communication)
            new_release_download[residual_computation<0.0001]=0
            next_time_slot=t+1
            delay_log.append(residual_buffer-1)
            residual_buffer-=1
            residual_buffer[residual_buffer<0]=0
            residual_buffer=residual_buffer+new_buffer
            #message='At time %i, the bit decision is (%.2f,%.2f,%.2f)'% (t,current_dec_bit[0][0],current_dec_bit[1][0],current_dec_bit[2][0])
            #message+='the cp decision is (%.2f,%.2f,%.2f)'% (t,current_dec_bit[0][0],current_dec_bit[1][0],current_dec_bit[2][0])
            print('At time %i, the decision is (%.2f,%.2f,%.2f),(%.2f,%.2f,%.2f),(%.2f,%.2f,%.2f), the residual is (%.2f,%.2f,%.2f) and (%.2f,%.2f,%.2f)' % (t,current_dec_bit[0][0],current_dec_bit[1][0],current_dec_bit[2][0],current_dec_cp[0][0],current_dec_cp[1][0],current_dec_cp[2][0],current_dec_bd[0][0],current_dec_bd[1][0],current_dec_bd[2][0],residual_computation[0][0],residual_computation[1][0],residual_computation[2][0],residual_communication[0][0],residual_communication[1][0],residual_communication[2][0]))
            print('At time %i, the downlink rate is (%.2f,%.2f,%.2f) and the new relesead buffer is (%.2f,%.2f,%.2f).'%(t,downlink_rate[0],downlink_rate[1],downlink_rate[2],new_buffer[0],new_buffer[1],new_buffer[2]))
            downlink_log.append(downlink_rate)
            cp_log.append([current_dec_cp[0][0],current_dec_cp[1][0],current_dec_cp[2][0]])
            #print(current_dec_bit)
            #print(current_dec_cp)
            #print(current_dec_bd)
            #print(residual_computation)
            #print(residual_communication)
            bitrate_log.append(current_dec_bit)
            current_dec_bit=np.zeros((num_of_user,1))
            new_buffer=np.zeros((num_of_user,1))
            #new_release_download=np.zeros((num_of_user,1))
            
    else:
        #print("I'm here!")
        #user_id=np.argmin(residual_computation+residual_communication)[0]
        #for i in range(4):
        #temp_residual_computation=residual_computation+np.reshape(one_hot_vector[user_id]*(10-candidate_bitrate[i]),(num_of_user,1))
        if sum(np.sqrt(residual_computation))<0.001:
            decision_computation=np.zeros((num_of_user,1))
        else:
            decision_computation=computation_upper_bound*(np.sqrt(residual_computation)/sum(np.sqrt(residual_computation)))
        #temp_residual_communication=residual_communication+np.reshape(one_hot_vector[user_id]*candidate_bitrate[i]*beta,(num_of_user,1))
        temp_residual_communication=residual_communication+new_release_download
        try:
            user_id=np.argmax(temp_residual_communication)[0]
        except:
            user_id=np.argmax(temp_residual_communication)
        #user_id=np.argmax(residual_computation+residual_communication)[0]
        
        weight=0
        for j in range(num_of_user):
            weight+=lamda[user_id]*np.sqrt(temp_residual_communication)[j]/(lamda[j]*np.sqrt(temp_residual_communication)[user_id])
        unit_bandwidth=(bandwith_upper_bound+(weight-num_of_user)/2)/weight
        decision_bandwidth=[]
        for j in range(num_of_user):
            decision_bandwidth.append(unit_bandwidth*lamda[user_id]*np.sqrt(temp_residual_communication)[j]/(lamda[j]*np.sqrt(temp_residual_communication)[user_id])-(lamda[user_id]*np.sqrt(temp_residual_communication)[j]/(lamda[j]*np.sqrt(temp_residual_communication)[user_id])-1)/2)
        #downlink_rate=decision_bandwidth*np.log2(1+np.reshape(lamda,(num_of_user,1))/np.array(decision_bandwidth))

        current_dec_cp=decision_computation
        current_dec_bd=decision_bandwidth

        
        residual_computation=residual_computation-current_dec_cp
        residual_computation[residual_computation<0]=0
        downlink_rate=np.array(current_dec_bd)*np.log2(1+np.reshape(lamda,(num_of_user,1))/np.dot(1000000,np.array(current_dec_bd)))
        downlink_rate[np.isnan(downlink_rate)]=0
        if sum(current_dec_cp)<0.01:
            #There is no new relesea
            old_rc=residual_communication
            residual_communication=residual_communication-downlink_rate
            #print(residual_communication)
            residual_communication[residual_communication<0]=0
            for user in range(num_of_user):
                    if old_rc[user]>0.01 and residual_communication[user]<0.01:
                        new_buffer[user]+=beta
        else:
            download_time_for_new_release=1-residual_computation/current_dec_cp
            download_time_for_new_release[np.isnan(download_time_for_new_release)]=0
            download_time_for_new_release[download_time_for_new_release<0]=0
            #print(residual_communication)
            residual_communication=residual_communication-(1-download_time_for_new_release)*downlink_rate
            #print(residual_communication)
            residual_communication[residual_communication<0]=0
            #print(residual_communication)
            for user in range(num_of_user):
                    if residual_computation[user]<0.0001:
                        residual_communication=residual_communication+np.reshape(one_hot_vector[user]*new_release_download[user],(num_of_user,1))
            #print(residual_communication)
            old_rc=residual_communication
            residual_communication=residual_communication-(download_time_for_new_release)*downlink_rate
            #print(residual_communication)
            residual_communication[residual_communication<0]=0
            for user in range(num_of_user):
                    if old_rc[user]>0.01 and residual_communication[user]<0.01:
                        new_buffer[user]+=beta
            #print(residual_communication)
            new_release_download[residual_computation<0.0001]=0
        
        next_time_slot=t+1
        
        delay_log.append(residual_buffer-1)
        residual_buffer-=1
        residual_buffer[residual_buffer<0]=0
        residual_buffer=residual_buffer+new_buffer
        print('At time %i, the decision is (%.2f,%.2f,%.2f),(%.2f,%.2f,%.2f),(%.2f,%.2f,%.2f), the residual is (%.2f,%.2f,%.2f) and (%.2f,%.2f,%.2f)' % (t,current_dec_bit[0][0],current_dec_bit[1][0],current_dec_bit[2][0],current_dec_cp[0][0],current_dec_cp[1][0],current_dec_cp[2][0],current_dec_bd[0][0],current_dec_bd[1][0],current_dec_bd[2][0],residual_computation[0][0],residual_computation[1][0],residual_computation[2][0],residual_communication[0][0],residual_communication[1][0],residual_communication[2][0]))
        print('At time %i, the downlink rate is (%.2f,%.2f,%.2f) and the new relesead buffer is (%.2f,%.2f,%.2f).'%(t,downlink_rate[0],downlink_rate[1],downlink_rate[2],new_buffer[0],new_buffer[1],new_buffer[2]))
        downlink_log.append(downlink_rate)
        cp_log.append([current_dec_cp[0][0],current_dec_cp[1][0],current_dec_cp[2][0]])
        bitrate_log.append(current_dec_bit)
        new_buffer=np.zeros((num_of_user,1))
        #print(current_dec_bit)
        #print(current_dec_cp)
        ##print(current_dec_bd)
        #print(residual_computation)
        #print(residual_communication)
        #new_release_download=np.zeros((num_of_user,1))

