import tkinter
from tkinter import *
import information

class Team:
    def __init__(self):
        self.root = Tk()
        self.root.config(background="light blue")
        self.root.wm_title("队伍管理系统")
        self.root.geometry("900x900")

        self.team_label = \
            tkinter.Label(self.root, text="所有队伍信息：", bg="tan", fg="black", font=("Fangsong", 15))
        self.team_text = Text(self.root, height=30, width=90, bg="cornsilk", fg="black")

        self.select_team_label = tkinter.Label(self.root, text="输入队伍名字", bg="tan", fg="black", font=("Fangsong", 15))
        self.select_team = Entry(self.root, width=20, bg="cornsilk")
        self.select_team_text = Text(self.root, height=30, width=90, bg="cornsilk", fg="black")
        self.course_button1 = Button(self.root, text="查询", bg="steel blue", fg="black", font=("KaiTi", 12),
                                    command=self.view_one_team)

        self.update_player_name_label = \
            tkinter.Label(self.root, text="选手ID", bg="tan", fg="black", font=("KaiTi", 15))
        self.update_player_position_label = \
            tkinter.Label(self.root, text="选手位置", bg="tan", fg="black", font=("KaiTi", 15))
        self.update_player_team_label = \
            tkinter.Label(self.root, text="选手队伍", bg="tan", fg="black", font=("KaiTi", 15))
        self.update_player_name_text = Entry(self.root, width=60, bg="cornsilk")
        self.update_player_position_text = Entry(self.root, width=40, bg="cornsilk")
        self.update_player_team_text = Entry(self.root, width=40, bg="cornsilk")

        self.old_player_name_label = \
            tkinter.Label(self.root, text="原选手ID:", bg="tan", fg="black", font=("KaiTi", 15))
        self.new_player_name_label = \
            tkinter.Label(self.root, text="改为ID:", bg="tan", fg="black", font=("KaiTi", 15))

        self.old_player_name_text = Entry(self.root, width=60, bg="cornsilk")
        self.new_player_name_text = Entry(self.root, width=40, bg="cornsilk")

        self.course_button2 = Button(self.root, text="签约", bg="steel blue", fg="black", font=("KaiTi", 12),
                                    command=self.contract_player)
        self.course_button3 = Button(self.root, text="改名", bg="steel blue", fg="black", font=("KaiTi", 12),
                                     command=self.rechristen_player)

        self.insert_team_name_label = \
            tkinter.Label(self.root, text="添加的队伍名字", bg="tan", fg="black", font=("KaiTi", 15))
        self.insert_team_top_label = \
            tkinter.Label(self.root, text="添加队伍上单", bg="tan", fg="black", font=("KaiTi", 15))
        self.insert_team_jug_label = \
            tkinter.Label(self.root, text="添加队伍打野", bg="tan", fg="black", font=("KaiTi", 15))
        self.insert_team_mid_label = \
            tkinter.Label(self.root, text="添加队伍中单", bg="tan", fg="black", font=("KaiTi", 15))
        self.insert_team_adc_label = \
            tkinter.Label(self.root, text="添加队伍ADC", bg="tan", fg="black", font=("KaiTi", 15))
        self.insert_team_sup_label = \
            tkinter.Label(self.root, text="添加队伍辅助", bg="tan", fg="black", font=("KaiTi", 15))
        self.insert_team_name_text = Entry(self.root,width=60, bg="cornsilk")
        self.insert_team_top_text = Entry(self.root, width=60, bg="cornsilk")
        self.insert_team_jug_text = Entry(self.root, width=60, bg="cornsilk")
        self.insert_team_mid_text = Entry(self.root, width=60, bg="cornsilk")
        self.insert_team_adc_text = Entry(self.root, width=60, bg="cornsilk")
        self.insert_team_sup_text = Entry(self.root, width=60, bg="cornsilk")
        self.course_button4 = Button(self.root, text="添加该队伍", bg="steel blue", fg="black", font=("KaiTi", 12),
                                     command=self.insert_team)

        self.delete_team_name_label = \
            tkinter.Label(self.root, text="待删除的队伍的名字", bg="tan", fg="black", font=("KaiTi", 15))
        self.delete_team_name_text = Entry(self.root, width=60, bg="cornsilk")
        self.course_button5 = Button(self.root, text="删除该队伍", bg="steel blue", fg="black", font=("KaiTi", 12),
                                     command=self.delete_team)

        self.old_team_name_label = tkinter.Label(self.root, text="原队名", bg="tan", fg="black", font=("KaiTi", 15))
        self.old_team_name = Entry(self.root, width=60, bg="cornsilk")
        self.new_team_name_label = tkinter.Label(self.root, text="新队名", bg="tan", fg="black", font=("KaiTi", 15))
        self.new_team_name = Entry(self.root, width=60, bg="cornsilk")
        self.course_button6 = Button(self.root, text="更改队名", bg="steel blue", fg="black", font=("KaiTi", 12),
                                     command=self.change_team_name)


    def inilize(self):
        self.team_label.grid(row=0, column=0, sticky=N)
        self.team_text.grid(row=2, column=0,sticky=N)

        self.select_team_label.grid(row=0, column=1, sticky=N)
        self.select_team.grid(row=1, column=1, sticky=N)
        self.select_team_text.grid(row=2, column=1, sticky=N)
        self.course_button1.grid(row=1, column=2, sticky=N)

        self.update_player_name_label.grid(row=3, column=0, sticky=N)
        self.update_player_name_text.grid(row=4, column=0, sticky=N)
        self.update_player_position_label.grid(row=3, column=1, sticky=N)
        self.update_player_position_text.grid(row=4, column=1, sticky=N)
        self.update_player_team_label.grid(row=3, column=2, sticky=N)
        self.update_player_team_text.grid(row=4, column=2, sticky=N)
        self.course_button2.grid(row=4, column=3, sticky=N)

        self.old_player_name_label.grid(row=5, column=0, sticky=N)
        self.old_player_name_text.grid(row=6, column=0, sticky=N)
        self.new_player_name_label.grid(row=5, column=1, sticky=N)
        self.new_player_name_text.grid(row=6, column=1, sticky=N)
        self.course_button3.grid(row=6, column=2, sticky=N)

        self.insert_team_name_label.grid(row=7, column=0, sticky=N)
        self.insert_team_top_label.grid(row=8, column=0, sticky=N)
        self.insert_team_jug_label.grid(row=9, column=0, sticky=N)
        self.insert_team_mid_label.grid(row=10, column=0, sticky=N)
        self.insert_team_adc_label.grid(row=11, column=0, sticky=N)
        self.insert_team_sup_label.grid(row=12, column=0, sticky=N)
        self.insert_team_name_text.grid(row=7, column=1, sticky=N)
        self.insert_team_top_text.grid(row=8, column=1, sticky=N)
        self.insert_team_jug_text.grid(row=9, column=1, sticky=N)
        self.insert_team_mid_text.grid(row=10, column=1, sticky=N)
        self.insert_team_adc_text.grid(row=11, column=1, sticky=N)
        self.insert_team_sup_text.grid(row=12, column=1, sticky=N)
        self.course_button4.grid(row=7, column=2, sticky=N)

        self.delete_team_name_label.grid(row=15, column=0, sticky=N)
        self.delete_team_name_text.grid(row=15, column=1, sticky=N)
        self.course_button5.grid(row=15, column=2, sticky=N)

        self.old_team_name_label.grid(row=16, column=0, sticky=N)
        self.old_team_name.grid(row=17, column=0, sticky=N)
        self.new_team_name_label.grid(row=16, column=1, sticky=N)
        self.new_team_name.grid(row=17, column=1, sticky=N)
        self.course_button6.grid(row=17, column=2, sticky=N)

# 添加队伍
    def insert_team(self):
        team_name = self.insert_team_name_text.get()
        top = self.insert_team_top_text.get()
        jug = self.insert_team_jug_text.get()
        mid = self.insert_team_mid_text.get()
        adc = self.insert_team_adc_text.get()
        sup = self.insert_team_sup_text.get()
        information.insert_team(team_name, top, jug, mid, adc, sup)
        self.update_ui()

#更改队名
    def change_team_name(self):
        old_name = self.old_team_name.get()
        new_name = self.new_team_name.get()
        information.update_team_name(old_name, new_name)
        self.update_ui()

   #删除某一只队伍
    def delete_team(self):
        team_name = self.delete_team_name_text.get()
        information.delete_team(team_name)
        self.update_ui()
#看某一只队伍
    def view_one_team(self):
        team_name = self.select_team.get()
        self.select_team_text.delete(1.0, END)
        information.display_team(self.select_team_text, team_name)
        self.update_ui()

#     签约
    def contract_player(self):
        player_id = self.update_player_name_text.get()
        player_pos = self.update_player_position_text.get()
        team_name = self.update_player_team_text.get()
        information.update_team_player(team_name, player_id, player_pos)
        self.update_ui()

#     选手改名
    def rechristen_player(self):
        old_id = self.old_player_name_text.get()
        new_id = self.new_player_name_text.get()
        information.update_player_name(old_id, new_id)

#     更新ui
    def update_ui(self):
        self.team_text.delete(1.0, END)
        information.display_team_all(self.team_text)

    def start(self):
        self.inilize()
        self.update_ui()
        self.root.mainloop()

if __name__=='__main__':
    team = Team()
    team.start()
