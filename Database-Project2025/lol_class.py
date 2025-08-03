import tkinter
from tkinter import *
import team_class
import information

class Coagame:
    def __init__(self):
        self.root = Tk()
        self.root.config(background="light blue")
        self.root.wm_title("LOL赛事")
        self.root.geometry("900x900")

        self.competition_all_label = tkinter.Label(self.root, text="所有赛事的信息",  bg="tan", fg="black", font=("KaiTi", 15))
        self.competition_select_label = tkinter.Label(self.root, text="所选赛事的信息", bg="tan", fg="black",
                                                   font=("KaiTi", 15))
        self.competition_insert_name_label = tkinter.Label(self.root, text="添加赛事名字", bg="tan", fg="black",
                                                   font=("KaiTi", 15))
        self.competition_insert_mvp_label = tkinter.Label(self.root, text="添加该赛事的FMVP", bg="tan", fg="black",
                                                   font=("KaiTi", 15))
        self.competition_insert_team_label = tkinter.Label(self.root, text="添加该赛事获胜队伍", bg="tan", fg="black",
                                                   font=("KaiTi", 15))


        self.competition_all_text = Text(self.root, height=13, width=60, bg="cornsilk", fg="black")
        self.competition_select_text = Text(self.root, height=13, width=60, bg="cornsilk", fg="black")

        self.competition_select = Entry(self.root, width=10, bg="cornsilk")
        self.competition_insert_name = Entry(self.root, width=10, bg="cornsilk")
        self.competition_insert_mvp = Entry(self.root, width=10, bg="cornsilk")
        self.competition_insert_team = Entry(self.root, width=10, bg="cornsilk")

        self.course_button1 = Button(self.root, text="选择", bg="steel blue", fg="black", font=("KaiTi", 12),
                                    command=self.select_competition)
        self.course_button2 = Button(self.root, text="添加", bg="steel blue", fg="black", font=("KaiTi", 12),
                                     command=self.add_competition)

        self.course_button3 = Button(self.root, text="删除异常的信息", bg="steel blue", fg="black", font=("KaiTi", 20),
                                     command=self.delete_wrong)

    def inilize(self):
        self.competition_all_label.grid(row=0, column=0, sticky=W)
        self.competition_all_text.grid(row=2, column=0, sticky=W)

        self.competition_select_label.grid(row=0, column=1, sticky=W)
        self.competition_select.grid(row=1, column=1, sticky=W)
        self.competition_select_text.grid(row=2, column=1, sticky=W)
        self.course_button1.grid(row=1, column=2, sticky=W)

        self.competition_insert_name_label.grid(row=3, column=0, sticky=W)
        self.competition_insert_name.grid(row=4, column=0, sticky=W)
        self.competition_insert_mvp_label.grid(row=3, column=1, sticky=W)
        self.competition_insert_mvp.grid(row=4, column=1, sticky=W)
        self.competition_insert_team_label.grid(row=3, column=2, sticky=W)
        self.competition_insert_team.grid(row=4, column=2, sticky=W)
        self.course_button1.grid(row=1, column=3, sticky=W)
        self.course_button2.grid(row=4, column=3, sticky=W)

        self.course_button3.grid(row=5, column=0, sticky=W)

#删除异常的信息
    def delete_wrong(self):
        information.delete_wrong_com()
        self.update_ui()
#添加赛事
    def add_competition(self):
        insert_name = self.competition_insert_name.get()
        insert_mvp = self.competition_insert_mvp.get()
        insert_champion = self.competition_insert_team.get()
        information.insert_competition(com_name=insert_name, mvp=insert_mvp, champion=insert_champion)
        self.update_ui()

#查询某一个赛事的信息
    def select_competition(self):
        self.competition_select_text.delete(1.0, END)
        com_name = self.competition_select.get()
        information.display_competition_info(self.competition_select_text, com_name)
        self.update_ui()
    def update_ui(self):
        self.competition_all_text.delete(1.0, END)
        information.display_competition_all(self.competition_all_text)

    def start(self):
        self.inilize()
        self.update_ui()
        self.root.mainloop()

if __name__=='__main__':
    coa = Coagame()
    coa.start()
