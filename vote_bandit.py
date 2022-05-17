# coding: utf-8
# Author: crb
# Date: 2021/7/17 20:53
import csv
import datetime
import logging
import pickle
import random
import sys
import os
import subprocess
import time
from collections import Counter
import numpy as np
from numpy.linalg import inv

from get_arm import bin_search, get_llc_bandwith_config, list_duplicates
from get_config import gen_config, gen_init_config, LC_APP_NAMES


class LinUCB():

    def __init__(self, ndims, alpha, app_id, core_narms=9, llc_narms=55, band_namrms=10):
        self.num_app = len(app_id)
        self.app_id = app_id

        self.core_narms = core_narms
        self.llc_narms = llc_narms
        self.band_namrms = band_namrms
        # number of context features
        self.ndims = ndims
        # explore-exploit parameter
        self.alpha = alpha

        self.A_c = {}
        self.b_c = {}
        self.p_c_t = {}

        self.A_l = {}
        self.b_l = {}
        self.p_l_t = {}

        self.A_b = {}
        self.b_b = {}
        self.p_b_t = {}

        for i in app_id:
            self.A_c[i] = np.zeros((self.core_narms, self.ndims * 2, self.ndims * 2))
            self.b_c[i] = np.zeros((self.core_narms, self.ndims * 2, 1))
            self.p_c_t[i] = np.zeros(self.core_narms)

            self.A_l[i] = np.zeros((self.llc_narms, self.ndims * 2, self.ndims * 2))
            self.b_l[i] = np.zeros((self.llc_narms, self.ndims * 2, 1))
            self.p_l_t[i] = np.zeros(self.llc_narms)

            self.A_b[i] = np.zeros((self.band_namrms, self.ndims * 2, self.ndims * 2))
            self.b_b[i] = np.zeros((self.band_namrms, self.ndims * 2, 1))
            self.p_b_t[i] = np.zeros(self.band_namrms)
            for arm in range(self.core_narms):
                self.A_c[i][arm] = np.eye(self.ndims * 2)

            for arm in range(self.llc_narms):
                self.A_l[i][arm] = np.eye(self.ndims * 2)

            for arm in range(self.band_namrms):
                self.A_b[i][arm] = np.eye(self.ndims * 2)

        super().__init__()
        return

    def add_del_app(self, app_id):
        A_c, A_l, A_b = 0, 0, 0
        for i in self.A_c.keys():
            A_c += self.A_c[i]
            A_l += self.A_l[i]
            A_b += self.A_b[i]

        for i in app_id:
            if i not in self.A_c.keys():
                self.A_c[i] = np.zeros((self.core_narms, self.ndims * 2, self.ndims * 2))
                self.b_c[i] = np.zeros((self.core_narms, self.ndims * 2, 1))
                self.p_c_t[i] = np.zeros((self.core_narms))
                for arm in range(self.core_narms):
                    self.A_c[i][arm] = A_c[arm] / self.num_app

                self.A_l[i] = np.zeros((self.llc_narms, self.ndims * 2, self.ndims * 2))
                self.b_l[i] = np.zeros((self.llc_narms, self.ndims * 2, 1))
                self.p_l_t[i] = np.zeros((self.llc_narms))
                for arm in range(self.llc_narms):
                    self.A_l[i][arm] = A_l[arm] / self.num_app

                self.A_b[i] = np.zeros((self.band_namrms, self.ndims * 2, self.ndims * 2))
                self.b_b[i] = np.zeros((self.band_namrms, self.ndims * 2, 1))
                self.p_b_t[i] = np.zeros((self.band_namrms))
                for arm in range(self.band_namrms):
                    self.A_b[i][arm] = A_b[arm] / self.num_app

        self.num_app = len(app_id)
        self.app_id = app_id

    def play(self, context, other_context, times):
        assert len(context[self.app_id[0]]) == self.ndims, 'the shape of context size is wrong'
        llc_action = {}
        band_action = {}
        contexts = {}
        # gains per each arm
        # only calculate the app in this colocation
        for key in self.app_id:
            A = self.A_c[key]
            b = self.b_c[key]
            contexts[key] = np.hstack((context[key], other_context[key]))

            for i in range(self.core_narms):
                # initialize theta hat
                theta = inv(A[i]).dot(b[i])
                # get context of each arm from flattened vector of length 100
                cntx = np.array(contexts[key])
                # get gain reward of each arm
                self.p_c_t[key][i] = theta.T.dot(cntx) + self.alpha * np.sqrt(cntx.dot(inv(A[i]).dot(cntx)))

            A = self.A_l[key]
            b = self.b_l[key]
            for i in range(self.llc_narms):
                theta = inv(A[i]).dot(b[i])
                cntx = np.array(contexts[key])
                self.p_l_t[key][i] = theta.T.dot(cntx) + self.alpha * np.sqrt(cntx.dot(inv(A[i]).dot(cntx)))

            llc_action[key] = np.random.choice(np.where(self.p_l_t[key] == max(self.p_l_t[key]))[0])

            A = self.A_b[key]
            b = self.b_b[key]
            for i in range(self.band_namrms):
                theta = inv(A[i]).dot(b[i])
                cntx = np.array(contexts[key])
                self.p_b_t[key][i] = theta.T.dot(cntx) + self.alpha * np.sqrt(cntx.dot(inv(A[i]).dot(cntx)))
            band_action[key] = np.random.choice(np.where(self.p_b_t[key] == max(self.p_b_t[key]))[0])

        core_action = beam_search(self.core_narms, self.app_id, self.p_c_t, times, end_condition=30)
        return core_action, llc_action, band_action

    def update(self, core_arms, llc_arms, band_arms, reward, context, other_context):
        contexts = {}
        for key in self.app_id:
            arm = core_arms[key]

            contexts[key] = np.hstack((context[key], other_context[key]))

            self.A_c[key][arm] += np.outer(np.array(contexts[key]),
                                           np.array(contexts[key]))

            self.b_c[key][arm] = np.add(self.b_c[key][arm].T,
                                        np.array(contexts[key]) * reward).reshape(
                self.ndims * 2, 1)

            arm = llc_arms[key]
            self.A_l[key][arm] += np.outer(np.array(contexts[key]),
                                           np.array(contexts[key]))
            self.b_l[key][arm] = np.add(self.b_l[key][arm].T,
                                        np.array(contexts[key]) * reward).reshape(
                self.ndims * 2, 1)

            arm = band_arms[key]
            self.A_b[key][arm] += np.outer(np.array(contexts[key]),
                                           np.array(contexts[key]))
            self.b_b[key][arm] = np.add(self.b_b[key][arm].T,
                                        np.array(contexts[key]) * reward).reshape(
                self.ndims * 2, 1)


def train_success(rounds=30):
    nof_counters = 22
    nof_colocation = len(colocation_list)

    init_alg = "fair"

    alpha = 0.01
    # number of bandit versions
    mab_num = 3
    mab_1 = LinUCB(nof_counters, alpha, colocation_list[0])
    mab_2, mab_3 = 0, 0
    mab_list = [mab_1, mab_2, mab_3]
    mab_count = 1
    tmp_cumulative_reward, tmp_G = [[] for _ in range(mab_num)], [0 for _ in range(mab_num)]
    nrounds = 0
    for col_items in range(nof_colocation):
        app_id = colocation_list[col_items]

        lc_app, bg_app = [], []
        for i in app_id:
            if i in LC_APP_NAMES:
                lc_app.append(i)
            else:
                bg_app.append(i)

        chose_arm_storage = []
        reward_arms = []

        core_list, llc_config, mb_config, chosen_arms = gen_init_config(app_id, llc_arm_orders,
                                                                        alg=init_alg)

        for i in range(rounds):
            if nrounds % 60 == 0:
                if mab_count < mab_num:
                    mab_list[mab_count] = LinUCB(nof_counters, alpha, colocation_list[col_items])
                    mab_count += 1
                else:
                    mab_count = 1
                    mab_list[0] = LinUCB(nof_counters, alpha, colocation_list[col_items])

            nrounds += 1
            if i == 0:
                context, another_context, reward, p95_list = get_now_ipc(lc_app, bg_app, core_list,
                                                                         performamce_counters)
                mab_1.add_del_app(app_id)

                chose_arm_storage.append([core_list, llc_config, mb_config])
                reward_arms, chosen_arms, tmp_cumulative_reward[0], tmp_G[0] = onlineEvaluate(mab_1, reward,
                                                                                              reward_arms, chosen_arms,
                                                                                              tmp_cumulative_reward[0],
                                                                                              context, another_context,
                                                                                              tmp_G[0], i)


            else:
                tmp_chosen_arms = []
                for ii in range(len(mab_list)):
                    if mab_list[ii] != 0:
                        mab_list[ii].add_del_app(app_id)
                        reward_arms_1, chosen_arms_1, tmp_cumulative_reward[ii], tmp_G[ii] = onlineEvaluate(
                            mab_list[ii], reward,
                            reward_arms, chosen_arms,
                            tmp_cumulative_reward[ii],
                            context,
                            another_context,
                            tmp_G[ii], i)
                        tmp_chosen_arms.append(chosen_arms_1)

                chosen_arms = list_duplicates(tmp_chosen_arms, app_id)
                reward_arms = reward_arms_1

                core_list, llc_config, mb_config = gen_config(app_id, chosen_arms, llc_arm_orders,
                                                              mb_arm_orders)
                time.sleep(1)
                context, another_context, reward, p95_list = get_now_ipc(lc_app, bg_app, core_list,
                                                                         performamce_counters)
                chose_arm_storage.append([core_list, llc_config, mb_config])

        best_reward_id = np.argmax(reward_arms)
        best_config = chose_arm_storage[best_reward_id - 1]
        best_reward = reward_arms[best_reward_id]

        print(f"best config {best_config}, best reward {best_reward}")
        print(f"last config {core_list},{llc_config},{mb_config},{load_list}, last reward {reward}")

        print(f'Mean reward of LinUCB with alpha = {alpha} is: ', np.mean(reward_arms))


def onlineEvaluate(mab, reward, reward_arms, chosen_arms, cumulative_reward, context, another_context, G, sample_times):
    """
    :param mab:
    :param rewards: ipc/delay
    :param contexts: counter
    :param nrounds:
    :return:
    """

    mab.update(chosen_arms[0], chosen_arms[1], chosen_arms[2], reward, context, another_context)

    core_action, llc_action, band_action = mab.play(context, another_context, sample_times)

    reward_arms.append(reward)

    G += reward
    cumulative_reward.append(G)

    chosen_arms = [core_action, llc_action, band_action]

    return reward_arms, chosen_arms, cumulative_reward, G


if __name__ == "__main__":
    llc_arm_orders, mb_arm_orders = get_llc_bandwith_config()
    colocation_list = [['img-dnn', 'xapian', "masstree"]]
    performamce_counters = we_choose()
    train_success()
