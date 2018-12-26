'''
Created on 2018年12月26日

@author: 37434
'''
# -*- coding: utf-8 -*
import numpy as np
from random import random
from math import exp, tanh
import svmutil 


def rand(a, b):
    return (b-a)*random()+a


def sigmoid(x):
    return 1.0/(1.0+exp(-x))


def sigmoid_derivate(y):
    return y*(1-y)


def tan_h(a):
    return tanh(a)


def tanh_derivate(b):
    return 1-b**2


class BPNetwork:

    def __init__(self):
        """
        initial variables
        """
        self.in_n = 0
        self.hide_n = 0
        self.out_n = 0
        """
        initial output value
        """
        self.in_val = []
        self.hide_val = []
        self.out_val = []
        """
        initial threshold
        """
        self.hide_t = []
        self.out_t = []
        """
        initial weight
        """
        self.ih_w = []
        self.ho_w = []
        self.function = {
            'Sigmoid': sigmoid,
            'SigmoidDerivate': sigmoid_derivate,
            'Tanh': tan_h,
            'TanhDerivate': tanh_derivate}

    def create_nn(self, ni, nh, no, activate_fun):
        self.in_n = ni
        self.hide_n = nh
        self.out_n = no

        self.in_val = np.zeros(self.in_n)
        self.hide_val = np.zeros(self.hide_n)
        self.out_val = np.zeros(self.out_n)

        self.ih_w = np.zeros([self.in_n, self.hide_n])
        self.ho_w = np.zeros([self.hide_n, self.out_n])
        for i in range(ni):
            for h in range(nh):
                self.ih_w[i][h] = rand(0, 1)
        for h in range(nh):
            for j in range(no):
                self.ho_w[h][j] = rand(0, 1)

        self.hide_t = np.zeros(self.in_n)
        self.out_t = np.zeros(self.out_n)
        for h in range(nh):
            self.hide_t[h] = rand(0, 1)
        for j in range(no):
            self.out_t[j] = rand(0, 1)

        self.af = self.function[activate_fun]
        self.afd = self.function[activate_fun+'_derivate']

    def predict(self, x):
        for i in range(self.in_n):
            self.in_val[i] = x[i]

        for h in range(self.hide_n):
            total = 0
            for i in range(self.in_n):
                total += self.in_val[i]*self.ih_w[i][h]
            self.hide_val[h] = self.af(total-self.hide_t[h])

        for j in range(self.out_n):
            total = 0
            for h in range(self.hide_n):
                total += self.hide_val[h]*self.ho_w[h][j]
            self.out_val[j] = self.af(total-self.out_t[j])

    def back_propagation(self, x, y, lr):
        self.predict(x)
        out_grid = np.zeros(self.out_n)
        for j in range(self.out_n):
            out_grid[j] = (y[j]-self.out_val[j])*self.afd(self.out_val[j])

        hide_grid = np.zeros(self.hide_n)
        for h in range(self.hide_n):
            for j in range(self.out_n):
                hide_grid[h] += self.ho_w[j]*out_grid[j]
            hide_grid[h] = hide_grid[h]*self.afd(self.hide_val[h])

        for h in range(self.hide_n):
            for i in range(self.in_n):
                self.ih_w[h][i] += lr*hide_grid[h]*self.in_val[i]

        for j in range(self.out_n):
            for h in range(self.hide_n):
                self.ho_w[j][h] += lr*out_grid[j]*self.hide_val[h]

        for h in range(self.in_n):
            self.hide_t[h] -= lr*hide_grid[h]

        for j in range(self.out_n):
            self.out_t[j] -= lr*out_grid[j]

    def train_standard(self, data_in, data_out, lr=0.05):
        e_k = []
        for k in range(len(data_in)):
            x = data_in[k]
            y = data_out[k]
            self.back_propagation(x, y, lr)

            delta_y = 0.0
            for j in range(self.out_n):
                delta_y += (y[j]-self.out_val[j])**2
            e_k.append(delta_y/2)

        e = sum(e_k)/len(e_k)

        return e, e_k

    def predict_label(self, x):
        y = []

        for n in range(len(x)):
            self.predict(x[n])
            if self.out_val[n] > 0.5:
                y.append(1)
            else:
                y.append(0)

        return np.array(y)
