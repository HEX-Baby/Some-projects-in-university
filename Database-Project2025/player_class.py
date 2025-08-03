import tkinter
from tkinter import *
import team_class
import information

class Coagame:
    def __init__(self):
        self.root = Tk()
        self.root.config(background="light blue")
        self.root.wm_title("选手名单")
        self.root.geometry("900x900")

        self.all_player_label = tkinter.Label(self.root, text="所有选手信息",  bg="tan", fg="black", font=("KaiTi", 15))
        self.all_player_text = Text(self.root, height=50, width=30, bg="cornsilk", fg="black")

        self.star_player_label = tkinter.Label(self.root, text="明星选手信息",  bg="tan", fg="black", font=("KaiTi", 15))
        self.star_player_text = Text(self.root, height=50, width=30, bg="cornsilk", fg="black")
        self.course_button1 = Button(self.root, text="查看", bg="steel blue", fg="black", font=("KaiTi", 12),
                                     command=self.view_star_player)

        self.player_select_id_label = tkinter.Label(self.root, text="选手ID",  bg="tan", fg="black", font=("KaiTi", 15))
        self.player_select_id = Entry(self.root, width=30, bg="cornsilk")
        self.player_select_id_text = Text(self.root, height=50, width=30, bg="cornsilk", fg="black")
        self.course_button2 = Button(self.root, text="查找", bg="steel blue", fg="black", font=("KaiTi", 12),
                                     command=self.view_select_player)

        self.player_insert_id_label = tkinter.Label(self.root, text="输入选手ID",  bg="tan", fg="black", font=("KaiTi", 15))
        self.player_insert_pos_label = tkinter.Label(self.root, text="输入选手位置",  bg="tan", fg="black", font=("KaiTi", 15))
        self.player_insert_id = Entry(self.root, width=30, bg="cornsilk")
        self.player_insert_pos = Entry(self.root, width=30, bg="cornsilk")
        self.course_button3 = Button(self.root, text="添加", bg="steel blue", fg="black", font=("KaiTi", 12),
                                     command=self.insert_one_player)

        self.player_delete_id_label = tkinter.Label(self.root, text="输入待删除选手ID", bg="tan", fg="black",
                                                    font=("KaiTi", 15))
        self.player_delete_id = Entry(self.root, width=30, bg="cornsilk")
        self.course_button4 = Button(self.root, text="删除", bg="steel blue", fg="black", font=("KaiTi", 12),
                                     command=self.delete_one_player)

        self.player_old_id_label = tkinter.Label(self.root, text="原ID",  bg="tan", fg="black", font=("KaiTi", 15))
        self.player_new_id_label = tkinter.Label(self.root, text="新ID",  bg="tan", fg="black", font=("KaiTi", 15))
        self.player_old_id = Entry(self.root, width=30, bg="cornsilk")
        self.player_new_id = Entry(self.root, width=30, bg="cornsilk")
        self.course_button5 = Button(self.root, text="更改ID", bg="steel blue", fg="black", font=("KaiTi", 12),
                                     command=self.change_player_id)


    def inilize(self):
        self.all_player_label.grid(row=0, column=0, sticky=W)
        self.all_player_text.grid(row=2, column=0, sticky=W)

        self.star_player_label.grid(row=0, column=1, sticky=W)
        self.star_player_text.grid(row=2, column=1, sticky=W)
        self.course_button1.grid(row=0, column=2, sticky=W)

        self.player_select_id_label.grid(row=0, column=3, sticky=W)
        self.player_select_id.grid(row=1, column=3, sticky=W)
        self.course_button2.grid(row=1, column=4, sticky=W)
        self.player_select_id_text.grid(row=2, column=3, sticky=W)

        self.player_insert_id_label.grid(row=3, column=0, sticky=W)
        self.player_insert_pos_label.grid(row=3, column=1, sticky=W)
        self.player_insert_id.grid(row=4, column=0, sticky=W)
        self.player_insert_pos.grid(row=4, column=1, sticky=W)
        self.course_button3.grid(row=4, column=2, sticky=W)

        self.player_delete_id_label.grid(row=5, column=0, sticky=W)
        self.player_delete_id.grid(row=6, column=0, sticky=W)
        self.course_button4.grid(row=6, column=2, sticky=W)

        self.player_old_id_label.grid(row=7, column=0, sticky=W)
        self.player_old_id.grid(row=8, column=0, sticky=W)
        self.player_new_id_label.grid(row=7, column=1, sticky=W)
        self.player_new_id.grid(row=8, column=1, sticky=W)
        self.course_button5.grid(row=8, column=2, sticky=W)

#更改选手ID
    def change_player_id(self):
        old_id = self.player_old_id.get()
        new_id = self.player_new_id.get()
        information.update_player_name(old_id, new_id)
        self.update_ui()


#查询视图——流量选手
    def view_star_player(self):
        self.star_player_text.delete(1.0, END)
        information.display_star_players(self.star_player_text)
        self.update_ui()

    def view_select_player(self):
        player_id = self.player_select_id.get()
        self.player_select_id_text.delete(1.0, END)
        information.display_player(self.player_select_id_text, player_id)
        self.update_ui()

    def insert_one_player(self):
        player_id = self.player_insert_id.get()
        player_pos = self.player_insert_pos.get()
        information.insert_player(player_id, player_pos)
        self.update_ui()

    def delete_one_player(self):
        player_id = self.player_delete_id.get()
        information.delete_player(player_id)
        self.update_ui()

    def update_ui(self):
        self.all_player_text.delete(1.0, END)
        information.display_all_player(self.all_player_text)


    def start(self):
        self.inilize()
        self.update_ui()
        self.root.mainloop()

if __name__=='__main__':
    coa = Coagame()
    coa.start()
