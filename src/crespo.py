# 連続値を使わない(離散的な)強化学習
# アクション　シチュエーション（change_view 等）も一部使っていない物あり
# 補助的な行動考えていない
# 2タイ２

# catchgame と　train を複合させて作成している

import player11
import threading
from trash.dqn_agent import DQNAgent
import numpy as np

from collections import deque


class Crespo(player11.Player11, threading.Thread):
    def __init__(self):
        super(Crespo, self).__init__()
        self.name = "Crespo"

    def play_0(self):
        self.m_strCommand[self.m_iTime] = "(turn 0)"
        if self.checkInitialMode():
            if self.checkInitialMode():
                self.setKickOffPosition()
                command = \
                    "(move " + str(self.m_dKickOffX) + " " + str(self.m_dKickOffY) + ")"
                self.m_strCommand[self.m_iTime] = command
        else:
            print("send command")

            # (コマンド生成)===================
            self.m_strCommand[self.m_iTime] = "(dash 100)"
            # ==================================


if __name__ == "__main__":
    plays = []
    for i in range(4):
        p = Crespo()
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
