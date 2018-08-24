# 連続値を使わない(離散的な)強化学習
# アクション　シチュエーション（change_view 等）も一部使っていない物あり
# 補助的な行動考えていない
# 2タイ２

# catchgame と　train を複合させて作成している

import player11
import threading
from dqn_agent import DQNAgent
import numpy as np
from socket import *
import math

from collections import deque


class Play(player11.Player11, threading.Thread):
    def __init__(self):
        super(Play, self).__init__()
        self.name = "soccer"
        self.enable_actions = (0, 1, 2, 3, 4, 5, 6)
        self.reward = 0
        self.terminal = False
        self.screen = 0
        self.agent = DQNAgent(self.enable_actions, self.name)
        self.replay_memory_size = 1000
        self.train_stop_time = 6000

        # replay memory
        self.D = deque(maxlen=self.replay_memory_size)

    def play_0(self):
        self.m_strCommand[self.m_iTime] = "(turn 0)"
        # 強化学習の各パラメータの初期化
        if self.m_strPlayMode == "kick_off_l" or "kick_off_r":
            frame = 0
            loss = 0.0
            Q_max = 0.0
            self.reset()
            state_t_1, terminal, reward_t = self.observe()
        if self.checkInitialMode():
            if self.checkInitialMode():
                self.setKickOffPosition()
                command = \
                    "(move " + str(self.m_dKickOffX) + " " + str(self.m_dKickOffY) + ")"
                self.m_strCommand[self.m_iTime] = command
        else:
            print("send command")

            # 強化学習のコード(コマンド生成)===================
            state_t = state_t_1
            action_t = self.agent.select_action(state_t, self.agent.exploration)
            # コマンド、報酬の更新
            self.update(action_t)
            # パラメータの更新
            state_t_1, reward_t, terminal = self.observe()

            self.agent.store_experience(state_t, action_t, reward_t, state_t_1, terminal)

            self.agent.experience_replay()

            frame += 1
            loss += self.agent.current_loss
            Q_max += np.max(self.agent.Q_values(state_t))
            if self.m_iNumber == 10:
                print(self.reward)
            # ==================================

        if self.m_iTime == self.train_stop_time:
            self.agent.save_model()


#=========================================以下機械学習訓練用コード

    # action と situationを更新する
    def update(self, action):
        """
        action:
             0: (turn 0)
             1: (turn 60)
             2: (turn -60)
             3: (dash 100)
             4: (dash -100)
             5: (kick 100 0)
             6: (kick 50 0)
        """
        # コマンドの更新
        if action == self.enable_actions[0]:
            # do nothing
            self.m_strCommand[self.m_iTime] = "(turn 0)"
        elif action == self.enable_actions[1]:
            # turn right
            self.m_strCommand[self.m_iTime] = "(turn 60)"
        elif action == self.enable_actions[2]:
            self.m_strCommand[self.m_iTime] = "(turn -60)"
        elif action == self.enable_actions[3]:
            self.m_strCommand[self.m_iTime] = "(dash 100)"
        elif action == self.enable_actions[4]:
            self.m_strCommand[self.m_iTime] = "(dash -100)"
        elif action == self.enable_actions[5]:
            self.m_strCommand[self.m_iTime] = "(kick 100 0)"
        elif action == self.enable_actions[6]:
            self.m_strCommand[self.m_iTime] = "(kick 50 0)"

        # 報酬の更新(時間経過によるペナルティも加味)
        # 右チームの場合
        if self.m_strSide.startswith("r"):
            if self.m_strPlayMode == "goal r":
                self.reward += 100
                self.terminal = True
            elif self.m_strPlayMode == "goal l":
                self.reward -= 100
                self.terminal = True
            else:
                self.reward -= 2
                if self.m_dBallX< 0:
                    self.reward += 1
                else:
                    self.reward -= 1
        # 左チームの場合
        else:
            if self.m_strPlayMode == "goal r":
                self.reward -= 100
            elif self.m_strPlayMode == "goal l":
                self.reward += 100
            else:
                self.reward -= 2
                if self.m_dBallX < 0:
                    self.reward -= 1
                else:
                    self.reward += 1

    # situation を表現する関数
    def draw(self):
        if self.m_strSide.startswith("r"):
            side = 0
        else:
            side = 1
        self.screen = np.array([self.m_dX, self.m_dY, self.m_dNeck, self.m_dBallX, self.m_dBallY, side])


    def observe(self):
        self.draw()
        return self.screen, self.terminal, self.reward

    # 得点が入ったらリセット
    def reset(self):
        self.reward = 0
        self.terminal = False


if __name__ == "__main__":
    plays = []
    for i in range(4):
        p = Play()
        plays.append(p)
        teamname = str(p.__class__.__name__)
        if i < 2:
            teamname += "left"
        else:
            teamname += "right"
        plays[i].initialize((i % 2 + 1), teamname, "localhost", 6000)
        plays[i].start()

# 状態s一覧
#
# self.m_dX
# self.m_dY
# self.m_dNeck
# self.m_dBallX
# self.m_dBallY
# self.m_strSide
#
# 行動a一覧
# (turn 60)
# (turn -60)
# (dash 100)
# (dash -100)
# (kick 100 0)
# (kick 50 0)
