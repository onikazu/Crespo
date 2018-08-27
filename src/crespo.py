# 連続値を使わない(離散的な)強化学習
# アクション　シチュエーション（change_view 等）も一部使っていない物あり
# 補助的な行動考えていない
# 2タイ２ for　left

# catchgame と　train を複合させて作成している

import player11
import threading
import numpy as np

from collections import deque


class Crespo(player11.Player11, threading.Thread):
    def __init__(self):
        super(Crespo, self).__init__()
        self.name = "Crespo"
        # =============for machine learning
        # 入力値分割数
        self.num_digitized = 6
        self.goal_average_reward = 195  # この報酬を超えると学習終了（中心への制御なし）
        # 出力数
        self.action_num = 7
        self.action = 0
        self.actions = ("(turn 0)", "(turn 60)", "(turn -60)", "(dash 100)", "(dash -100)", "(kick 100 0)", "(kick 50 0)")
        self.max_number_of_steps = 100  # 1試行のstep数
        self.num_consecutive_iterations = 100  # 学習完了評価に使用する平均試行回数
        self.num_episodes = 2000  # 総試行回数
        self.q_table = np.random.uniform(low=-1, high=1, size=(self.num_digitized ** 5, self.action_num))
        self.total_reward_vec = np.zeros(self.num_consecutive_iterations)  # 各試行の報酬を格納
        self.final_x = np.zeros((self.num_episodes, 1))  # 学習後、各試行のt=200でのｘの位置を格納
        self.islearned = 0  # 学習が終わったフラグ
        self.isrender = 0  # 描画フラグ
        self.sum_reward = 0  # 報酬の総量

    # 実行
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
            self.m_strCommand[self.m_iTime] = self.after_play()
            # ==================================


    def after_play(self):
        self.m_strCommand = "(turn 0)"
        observation = (self.m_dX, self.m_dY, self.m_dBallX, self.m_dBallY, self.m_dNeck)
        state = self.digitize_state(observation)
        print(observation[0])  # プレーヤーのxイチを出力
        print(self.sum_reward)

        reward = 0
        # 報酬を決定
        if observation[2] > 20.0:
            reward = 1
        if self.m_strPlayMode == "(goal_l)":
            reward = 100

        self.sum_reward += reward

        # 離散状態s_{t+1}を求め、Q関数を更新する
        next_state = self.digitize_state(observation)  # t+1での観測状態を、離散値に変換
        self.q_table = self.update_Qtable(self.q_table, state, self.action, reward, next_state)

        #  次の行動a_{t+1}を求める
        episode = -(self.m_iTime // 100)
        self.action = self.get_action(next_state, episode)  # a_{t+1}
        return self.actions[self.action]


    # [1]Q関数を離散化して定義する関数　------------
    # 観測した状態を離散値にデジタル変換する
    def bins(self, clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    # 各値を離散値に変換
    def digitize_state(self, observation):
        cart_pos, cart_v, pole_angle, pole_v, a = observation
        digitized = [
            np.digitize(cart_pos, bins=self.bins(-52.5, 52.5, self.num_digitized)),  # dX
            np.digitize(cart_v, bins=self.bins(-34.0, 34.0, self.num_digitized)),  # dY
            np.digitize(pole_angle, bins=self.bins(-52.5, 52.5, self.num_digitized)),  # dBallX
            np.digitize(pole_v, bins=self.bins(-34.0, 34.0, self.num_digitized)),  # dBallY
            np.digitize(a, bins=self.bins(-180.0, 180.0, self.num_digitized))  # dNeck
        ]
        return sum([x * (self.num_digitized ** i) for i, x in enumerate(digitized)])

    # [2]行動a(t)を求める関数 -------------------------------------
    def get_action(self, next_state, episode):
        # 徐々に最適行動のみをとる、ε-greedy法
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0, 1):
            next_action = np.argmax(self.q_table[next_state])
        else:
            # action数は6個
            next_action = np.random.choice([0, 1, 2, 3, 4, 5])
        return next_action

    # [3]Qテーブルを更新する関数 -------------------------------------
    def update_Qtable(self, q_table, state, action, reward, next_state):
        gamma = 0.99
        alpha = 0.5
        next_Max_Q = max(q_table[next_state][0], q_table[next_state][1], q_table[next_state][2], q_table[next_state][3],
                         q_table[next_state][4], q_table[next_state][5], q_table[next_state][6])
        q_table[state, action] = (1 - alpha) * q_table[state, action] + \
                                 alpha * (reward + gamma * next_Max_Q)
        return q_table

    def analyzeMessage(self, message):
        # 初期メッセージの処理
        # print("p11:message:", message)
        if message.startswith("(init "):
            self.analyzeInitialMessage(message)
        # 視覚メッセージの処理
        elif message.startswith("(see "):
            self.analyzeVisualMessage(message)
        # 体調メッセージの処理
        elif message.startswith("(sense_body "):
            self.analyzePhysicalMessage(message)
            if self.m_iVisualTime < self.m_iTime:
                self.predict(self.m_iVisualTime, self.m_iTime)
            self.play_0()
            self.send(self.m_strCommand[self.m_iTime])
            self.after_play()
        # 聴覚メッセージの処理
        elif message.startswith("(hear "):
            self.analyzeAuralMessage(message)
        # サーバパラメータの処理
        elif message.startswith("(server_param"):
            self.analyzeServerParam(message)
        # プレーヤーパラメータの処理
        elif message.startswith("(player_param"):
            self.analyzePlayerParam(message)
        # プレーヤータイプの処理
        elif message.startswith("(player_type"):
            self.analyzePlayerType(message)
            # print("player_type_message", message)
        # エラーの処理
        else:
            print("p11 サーバーからエラーが伝えられた:", message)
            print("p11 エラー発生原因のコマンドは右記の通り :", self.m_strCommand[self.m_iTime])


if __name__ == "__main__":
    plays = []
    for i in range(4):
        p = Crespo()
        plays.append(p)
        teamname = str(p.__class__.__name__)
        if i < 11:
            teamname += "left"
        else:
            teamname += "right"
        plays[i].initialize((i % 2 + 1), teamname, "localhost", 6000)
        plays[i].start()

# 離散化させなくてはならない？(6分割**5変数の状態が生み出される)
# 状態s一覧
#
# self.m_dX
# self.m_dY
# self.m_dNeck
# self.m_dBallX
# self.m_dBallY
#
# 行動a一覧
# (turn 0)
# (turn 60)
# (turn -60)
# (dash 100)
# (dash -100)
# (kick 100 0)
# (kick 50 0)
