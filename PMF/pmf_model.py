from __future__ import print_function
import numpy as np
from numpy.random import RandomState
import pickle
import os
import copy
from evaluations import *
class PMF():
    '''
    a class for this Double Co-occurence Factorization model
    '''
    # initialize some paprameters
    def __init__(self, R, lambda_alpha=1e-2, lambda_beta=1e-2, latent_size=50, momuntum=0.8,
                 lr=0.001, iters=200, seed=None):
        #  超参数
        self.lambda_alpha = lambda_alpha
        self.lambda_beta = lambda_beta
        #  按一定比例保留之前的梯度
        self.momuntum = momuntum
        #  用户对电影的评分矩阵NxM
        self.R = R
        # ？？
        self.random_state = RandomState(seed)
        #  迭代次数
        self.iterations = iters
        #  学习率
        self.lr = lr
        #  指示函数，此处用矩阵表示，1表示用户对电影打分，0表示未打分
        self.I = copy.deepcopy(self.R)
        self.I[self.I != 0] = 1
        #  生成用户和电影的特征矩阵，U的维度是NxD,V的维度是DxM
        self.U = 0.1*self.random_state.rand(np.size(R, 0), latent_size)
        self.V = 0.1*self.random_state.rand(np.size(R, 1), latent_size)


    def loss(self):
        # the loss function of the model
        # 也就是论文中的目标函数E，目的是最小化loss
        loss = np.sum(self.I*(self.R-np.dot(self.U, self.V.T))**2) + self.lambda_alpha*np.sum(np.square(self.U)) + self.lambda_beta*np.sum(np.square(self.V))
        return loss
    def predict(self, data):
        # data是验证集，取用户和电影这两个维度
        index_data = np.array([[int(ele[0]), int(ele[1])] for ele in data], dtype=int)  # len(ele)x2维度
        '''
        self.U.take(index_data.take(0, axis=1), axis=0):根据用户id获得对应的U矩阵，
        '''
        u_features = self.U.take(index_data.take(0, axis=1), axis=0)  # U是NxD维度，index_data.take(0, axis=1)取的是用户信息
        v_features = self.V.take(index_data.take(1, axis=1), axis=0)  #
        '''
        axis= 0 对a的横轴进行操作，在运算的过程中其运算的方向表现为纵向运算,axis= 1 对a的纵轴进行操作，在运算的过程中其运算的方向表现为横向运算
        '''
        preds_value_array = np.sum(u_features*v_features, 1) # 计算预测的R,u_features*v_features是NxM维，横向求和之后是Nx1维度
        return preds_value_array

    def train(self, train_data=None, vali_data=None):
        '''
        # training process
        :param train_data: train data with [[i,j],...] and this indicates that K[i,j]=rating
        :param lr: learning rate
        :param iterations: number of iterations
        :return: learned V, T and loss_list during iterations
        '''
        train_loss_list = []
        vali_rmse_list = []
        last_vali_rmse = None

        # monemtum
        momuntum_u = np.zeros(self.U.shape)  # NxD维度
        momuntum_v = np.zeros(self.V.shape)  # DxM维度

        for it in range(self.iterations):
            # 梯度下降
            # derivate of Vi，U的梯度，整个矩阵
            grads_u = np.dot(self.I*(self.R-np.dot(self.U, self.V.T)), -self.V) + self.lambda_alpha*self.U

            # derivate of Tj，V的梯度，整个矩阵
            grads_v = np.dot((self.I*(self.R-np.dot(self.U, self.V.T))).T, -self.U) + self.lambda_beta*self.V

            # update the parameters
            '''
            momuntum_u、momuntum_v保存的是前一次迭代所得到的梯度，乘以self.momuntum表示按比例保留之前所得到的梯度，如果是同方向则是加速作用，如果是反方向则是缓冲作用
            '''
            momuntum_u = (self.momuntum * momuntum_u) + self.lr * grads_u
            momuntum_v = (self.momuntum * momuntum_v) + self.lr * grads_v
            # 更新U、V矩阵
            self.U = self.U - momuntum_u
            self.V = self.V - momuntum_v

            # training evaluation
            # 计算训练时的损失
            train_loss = self.loss()
            # 将训练时的损失保存在数组中
            train_loss_list.append(train_loss)
            # 输入验证集对模型进行预测，获得预测的R
            vali_preds = self.predict(vali_data)
            # 与真实的评分计算均方根误差
            vali_rmse = RMSE(vali_data[:,2], vali_preds)
            # 将每次的rmse保存到列表中
            vali_rmse_list.append(vali_rmse)

            print('traning iteration:{: d} ,loss:{: f}, vali_rmse:{: f}'.format(it, train_loss, vali_rmse))
            # 训练截止条件：last_vali_rmse不为空且rmse比前一次迭代的大或相等
            if last_vali_rmse and (last_vali_rmse - vali_rmse) <= 0:
                print('convergence at iterations:{: d}'.format(it))
                break
            else:
                last_vali_rmse = vali_rmse
        # 返回训练得到的U、V、loss_list、rmse_list
        return self.U, self.V, train_loss_list, vali_rmse_list
