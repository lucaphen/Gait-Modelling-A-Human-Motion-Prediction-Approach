#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from utils import h36motion as datasets
from utils.loss_funcs import euler_error
from utils.data_utils import expmap2xyz_torch,define_actions
import pandas as pd



data = []
# frames = np.arange(1, 217,1)
frames = np.arange(1, 49, 1)

def create_pose(ax,plots,vals,pred=True,update=False):    


    """
    (0,1)   =>  Left pelvis
    (0,6)   =>  Right pelvis

    // LEFT Lower Body
    (1,25)  =>  Left Spine
    (1,2)   =>  Upper Left Leg
    (2,3)   =>  Lower Left Leg
    (3,4)   =>  Left Foot

    // Right Lower Body
    (6,17)  =>  Right Spine
    (6,7)   =>  Upper Right Leg
    (7,8)   =>  Lower Right Leg
    (8,9)  =>  Right Foot

    """   

    # Dictionary of kinematic joints (in joint form, not vectors)
    """
    k_joints = {
        'spine' : [ k_tree[3][0][1], k_tree[3][1][1] ],
        'hip'   : [ k_tree[3][0][0], k_tree[3][1][0] ],
        'knee'  : [ k_tree[0][0][1], k_tree[0][1][1] ],
        'ankle' : [ k_tree[2][0][0], k_tree[2][1][0] ],
        'toe'   : [ k_tree[2][0][1], k_tree[2][1][1] ]  
    }
    
    
    """

    # h36m 32 joints(full)
    connect = [
             # (1, 2), 
             # (2, 3), 
             # (3, 4),  
             # (4, 5),   
             (6, 7), 
             (7, 8), 
             (8, 9), 
             # (9, 10),    
             # (0, 1), 
             (0, 6),     
            # (1, 25),
             (6, 17),   
            
            # Ignore these (Upper body)
            # (17, 18), (18, 19), (19, 20), (20, 21), (21, 22),
            # (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
            # (24, 25), (24, 17),
            # (24, 14), (14, 15)
    ]
    LR = [
            False, True, True, True, True,
            True, False, False, False, False,
            False, True, True, True, True,
            True, True, False, False, False,
            False, False, False, False, True,
            False, True, True, True, True,
            True, True
    ]  
    if pred:
        print("Predicted")
    else: 
        print("GT")
# Start and endpoints of our representation
    I   = np.array([touple[0] for touple in connect])
    J   = np.array([touple[1] for touple in connect])
    # print(len(I))
    # print(I)

# Left / right indicator
    LR  = np.array([LR[a] or LR[b] for a,b in connect])
    if pred:
        lcolor = "#C1292E"  # red
        rcolor = "#235789"  # red
    else:
        lcolor = "#235789"  # blue
        rcolor = "#235789"  # blue
    k_tree = []
    # print(vals)
    for i in np.arange( len(I) ):
        x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
        z = np.array( [vals[I[i], 1], vals[J[i], 1]] )
        y = np.array( [vals[I[i], 2], vals[J[i], 2]] )
        k_tree.append([y,z])

        # if i == 0: # This is 1,2 joint (spine, hip)
        #     x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
        #     z = np.array( [vals[I[i], 1], vals[J[i], 1]] )
        #     y = np.array( [vals[I[i], 2], vals[J[i], 2]] )
        #     # if len(data)%2==0: 
        #     #     data.append([y[1], z[1], y[1], 0])
        #     # else: 
        #     #     data.append([y[1], z[1], 0, y[1]])
        #     # print(data)
        #     print("Joint 1 (Hip): (Y: {0} , Z: {1})\nJoint 2 (Knee): (Y: {2} , Z: {3})".format(y[0],z[0], y[1], z[1]))
        # elif i==1:
        #     x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
        #     z = np.array( [vals[I[i], 1], vals[J[i], 1]] )
        #     y = np.array( [vals[I[i], 2], vals[J[i], 2]] )
        #     print("Joint 3 (Spine): (Y: {0} , Z: {1})".format(y[1],z[1]))
        # else: 
        #     x = np.array( [vals[I[i], 0], vals[J[i], 0]] )
        #     z = np.array( [vals[I[i], 1], vals[J[i], 1]] )
        #     y = np.array( [vals[I[i], 2], vals[J[i], 2]] )
        
        if not update:

            if i ==0:
                plots.append(ax.plot(x, y, z, lw=2,linestyle='--' ,c=lcolor if LR[i] else rcolor,alpha=1.0 if pred else 0, label=['GT' if not pred else 'Pred']))
            else:
                plots.append(ax.plot(x, y, z, lw=2,linestyle='--', c=lcolor if LR[i] else rcolor, alpha=1.0 if pred else 0 ))
        elif update:
            plots[i][0].set_xdata(x)
            plots[i][0].set_ydata(y)
            plots[i][0].set_3d_properties(z)
            plots[i][0].set_color(lcolor if LR[i] else rcolor)
            # plots[i][0].alpha()
    # print(k_tree)
    # print(len(data))
    # map k_tree to k_joints
    k_joints = {
        'spine' : [ k_tree[3][0][1], k_tree[3][1][1] ],
        'hip'   : [ k_tree[3][0][0], k_tree[3][1][0] ],
        'knee'  : [ k_tree[0][0][1], k_tree[0][1][1] ],
        'ankle' : [ k_tree[2][0][0], k_tree[2][1][0] ],
        'toe'   : [ k_tree[2][0][1], k_tree[2][1][1] ]  
    }
    
    k_joints_formatted = [k_joints['spine'][0], k_joints['spine'][1], k_joints['hip'][0], k_joints['hip'][1], k_joints['knee'][0],k_joints['knee'][1], k_joints['ankle'][0], k_joints['ankle'][1], k_joints['toe'][0], k_joints['toe'][1] ]
    data.append(k_joints_formatted)
    print(len(data))
    predicted_frames2 = 48
    predicted_frames25 = 216
    
    if len(data) == predicted_frames2: 
        print("Made it here")
        # Write data to excel
        df1 = pd.DataFrame(data,
            index=frames,
                columns=['Spine (y)', 'Spine (z)', 'Hip (y)', 'Hip (z)', 'Knee (y)','Knee (z)','Ankle (y)', 'Ankle (z)', 'Toe (y)', 'Toe (z)'])
        df1.to_excel("new_results_STS_GCN_2x.xlsx")

    return plots
   # ax.legend(loc='lower left')


# In[11]:


def update(num,data_gt,data_pred,plots_gt,plots_pred,fig,ax):
    
    gt_vals=data_gt[num]
    pred_vals=data_pred[num]
    plots_gt=create_pose(ax,plots_gt,gt_vals,pred=False,update=True)
    plots_pred=create_pose(ax,plots_pred,pred_vals,pred=True,update=True)
    
    

    
    
    r = 0.75
    xroot, zroot, yroot = gt_vals[0,0], gt_vals[0,1], gt_vals[0,2] # joint n 12 (back) as root
   # print(xroot,yroot,zroot)
    ax.set_xlim3d([-r+xroot, r+xroot])
    ax.set_ylim3d([-r+yroot, r+yroot])
    ax.set_zlim3d([-r+zroot, r+zroot])
    #ax.set_title('pose at time frame: '+str(num))
    #ax.set_aspect('equal')
 
    return plots_gt,plots_pred
    


# In[12]:


def visualize(input_n,output_n,visualize_from,path,modello,device,n_viz,skip_rate,actions):
    actions=define_actions(actions)
    actions = ["walking"]   

    for action in actions:
        if visualize_from=='train':
            loader=datasets.Datasets(path,input_n,output_n,skip_rate, split=0,actions=[action])
        elif visualize_from=='validation':
            loader=datasets.Datasets(path,input_n,output_n,skip_rate, split=1,actions=[action])
        elif visualize_from=='test':
            loader=datasets.Datasets(path,input_n,output_n,skip_rate, split=2,actions=[action])
            
        dim_used = np.array([6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41, 42,
                            43, 44, 45, 46, 47, 51, 52, 53, 54, 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85,
                            86])
                  
        loader = DataLoader(
        loader,
        batch_size=1,
        shuffle = True,
        num_workers=0)       

        data = []
        for cnt,batch in enumerate(loader):     
            print("Counter: {}".format(batch))
            batch = batch.to(device) 
            
            all_joints_seq=batch.clone()[:, input_n:input_n+output_n,:]
            
            sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3).permute(0,3,1,2)
            sequences_gt=batch[:, input_n:input_n+output_n, :]
            
            sequences_predict=modello(sequences_train).permute(0,1,3,2).contiguous().view(-1,output_n,len(dim_used))
            
            all_joints_seq[:,:,dim_used] = sequences_predict
            
            
            loss=euler_error(all_joints_seq,sequences_gt)# # both must have format (batch,T,V,C)

            all_joints_seq=all_joints_seq.view(-1,99)
            
            sequences_gt=sequences_gt.view(-1,99)
            
            all_joints_seq=expmap2xyz_torch(all_joints_seq).view(-1,output_n,32,3)
            
            sequences_gt=expmap2xyz_torch(sequences_gt).view(-1,output_n,32,3)

            

            

            data_pred=torch.squeeze(all_joints_seq,0).cpu().data.numpy()/1000 # in meters
            data_gt=torch.squeeze(sequences_gt,0).cpu().data.numpy()/1000


            fig = plt.figure()
            ax = Axes3D(fig)
            vals = np.zeros((32, 3)) # or joints_to_consider
            gt_plots=[]
            pred_plots=[]
        
            gt_plots=create_pose(ax,gt_plots,vals,pred=False,update=False)
            pred_plots=create_pose(ax,pred_plots,vals,pred=True,update=False)
            
            print("Length of Predicted Plots : {0}\nLength of GroundTruth Plots : {1}".format(len(gt_plots), len(pred_plots)) )


            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.legend(loc='lower left')

            ax.set_xlim3d([-1, 1.5])
            ax.set_xlabel('X')

            ax.set_ylim3d([-1, 1.5])
            ax.set_ylabel('Y')

            ax.set_zlim3d([0.0, 1.5])
            ax.set_zlabel('Z')
            ax.set_title('loss in euler angle is: '+str(round(loss.item(),4))+' for action : '+action+' for '+str(output_n)+' frames')

            line_anim = animation.FuncAnimation(fig, update, output_n, fargs=(data_gt,data_pred,gt_plots,pred_plots,
                                                                       fig,ax),interval=70, blit=False, repeat=False)
            plt.show()

            
            if cnt==n_viz-1:
                break

