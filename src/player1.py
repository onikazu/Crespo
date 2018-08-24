from socket import *
import threading
import sys

class Player1(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.socket = socket(AF_INET, SOCK_DGRAM)
        # self.socket2 = socket(AF_INET, SOCK_DGRAM)
        self.HOSTNAME = "localhost"
        self.PORT = 6000
        self.ADDRESS = gethostbyname(self.HOSTNAME)
        self.m_strPlayMode = ""
        self.m_iNumber = 0
        self.m_strTeamName = ""
        self.m_strHostName = ""
        self.m_strSide = ""
        self.m_debugLv01 = False

    def send(self, command):
        if len(command) == 0:
            return
        # v8 以降は専用ソケットが必要なので作成
        # if command.startswith("(init "):
        # print("p1command No", self.m_iNumber, command)

        # ヌル終端文字列の欠損による警告を防ぐ
        command = command + "\0"

        try:
            to_byte_command = command.encode(encoding='utf_8')
            self.socket.sendto(to_byte_command, (self.ADDRESS, self.PORT))
            # print("sending ", command, " is done")
            # else:
            #     to_byte_command = command.encode(encoding='utf_8')
            #     self.socket2.sendto(to_byte_command, (self.ADDRESS, self.PORT))
        except OSError:
            print("送信失敗")
            sys.exit()

    def receive(self):
        try:
            message, arr = self.socket.recvfrom(4096)
            message = message.decode("UTF-8")
            # ポート番号をｉｎｉｔに用いてモノから専用ソケットのものに変え無くてはならない！！！（重要）
            self.PORT = arr[1]
            # print("メッセージ（サーバーから", self.m_iNumber, "番）：", message)
            return message
        except OSError:
            print("受信失敗")
            sys.exit()
            return ""


    def initialize(self, number, team_name, server_name, server_port):
        self.m_iNumber = number
        self.m_strTeamName = team_name
        self.m_strHostName = server_name
        self.PORT = server_port
        # バージョンを指定しないと自動でバージョン３のプロトコルが動作する。
        if self.m_iNumber == 1:
            command = "(init " + self.m_strTeamName + "(goalie)(version 15.40))"
        else:
            command = "(init " + self.m_strTeamName + "(version 15.40))"
        self.send(command)

    # thread を動かしている最中に行われる関数
    def run(self):
        while True:
            message = self.receive()
            # print(message)
            self.analyzeMessage(message)

    def analyzeInitialMessage(self, message):
        index0 = message.index(" ")
        index1 = message.index(" ", index0 + 1)
        index2 = message.index(" ", index1 + 1)
        index3 = message.index(")", index2 + 1)

        self.m_strSide = message[index0+1:index1]
        self.m_iNumber = int(message[index1+1:index2])
        self.m_strPlayMode = message[index2+1:index3]

    def analyzeMessage(self, message):
        if isinstance(message, type(None)):
            return
            # print(message)
        elif message.startswith("(init"):
            self.analyzeInitialMessage(message)
        elif message.startswith("(warning") or message.startswith("(error"):
            print("Something is wrong when analyzeMessage: ", message)


if __name__ == "__main__":
    players = []
    for i in range(11):
        p = Player1()
        players.append(p)
        players[i].initialize(i+1, "kazu", "localhost", 6000)
        players[i].start()

    players[0].m_debugLv01 = True
    print("試合登録完了")
