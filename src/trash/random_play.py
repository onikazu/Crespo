# 連続値を使わない(離散的な)強化学習
# アクション　シチュエーション（change_view 等）も一部使っていない物あり
# 補助的な行動考えていない
# 2タイ２

# catchgame と　train を複合させて作成している

import player11
import threading
from socket import *
import math
import random

from collections import deque


class RandomPlayer(player11.Player11, threading.Thread):
    def __init__(self):
        super(RandomPlayer, self).__init__()
        self.name = "Random"
        self.enable_actions = (0, 1, 2, 3, 4, 5, 6)
        self.reward = 0
        self.screen = 0

    def play_0(self):
        self.m_strCommand[self.m_iTime] = "(turn 0)"
        if self.checkInitialMode():
            if self.checkInitialMode():
                self.setKickOffPosition()
                command = \
                    "(move " + str(self.m_dKickOffX) + " " + str(self.m_dKickOffY) + ")"
                self.m_strCommand[self.m_iTime] = command
        else:
            # (コマンド生成)===================
            choose = random.randint(0, 1, 2, 3, 4, 5, 6)
            self.m_strCommand[self.m_iTime] = self.enable_actions[choose]
            # ==================================
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
if __name__ == "__main__":
    plays = []
    for i in range(4):
        p = RandomPlayer()
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
