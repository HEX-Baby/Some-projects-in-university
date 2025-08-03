import pymysql
import numpy as np
from tkinter import *

# 添加选手
def insert_player(player_id, position):
    db = pymysql.connect(host='localhost', user='root', password='123456', database='lol', charset='utf8')
    cursor = db.cursor()
    sql = "INSERT INTO player VALUES (%s, %s)"
    val = (player_id, position)

    try:
        cursor.execute(sql, val)
        db.commit()
        print(f'成功添加选手{player_id}，其位置为{position}')
    except :
        db.rollback()  # 回滚到上一次提交的状态
        print("该选手已存在")

    # 关闭游标和数据库连接
    cursor.close()
    db.close()

# 添加队伍
def insert_team(team_name, top, jug, mid, adc, sup):
    db = pymysql.connect(host='localhost', user='root', password='123456', database='lol', charset='utf8')
    cursor = db.cursor()
    sql = "insert into team values (%s, %s, %s, %s, %s, %s)"
    val = (team_name, top, jug, mid, adc, sup)

    try:
        cursor.execute(sql, val)
        db.commit()
    except :
        db.rollback()  # 回滚到上一次提交的状态
        print("队伍添加失败。")

    # 关闭游标和数据库连接
    cursor.close()
    db.close()

# 添加新比赛
def insert_competition(com_name, mvp, champion):
    db = pymysql.connect(host='localhost', user='root', password='123456', database='lol', charset='utf8')
    cursor = db.cursor()
    sql = "insert into competition values (%s, %s, %s)"
    val = (com_name, mvp, champion)

    try:
        cursor.execute(sql, val)
        db.commit()
        print(f'已添加比赛：{com_name}，FMVP为{mvp}，冠军队伍为{champion}')
    except :
        db.rollback()  # 回滚到上一次提交的状态
        print("新一轮赛事添加失败！！！")

    # 关闭游标和数据库连接
    cursor.close()
    db.close()

# 删除选手
def delete_player(player_id):
    db = pymysql.connect(host='localhost', user='root', password='123456', database='lol', charset='utf8')
    cursor = db.cursor()
    sql = "delete from player where player_id = %s "
    val = (player_id,)
    try:
        cursor.execute(sql, val)
        db.commit()
        print(f'已删除选手：{player_id}')
    except :
        db.rollback()  # 回滚到上一次提交的状态
        print("该队员曾获得FMVP，无法删除队员！！！")

    # 关闭游标和数据库连接
    cursor.close()
    db.close()

# 删除队伍
def delete_team(team_name):
    db = pymysql.connect(host='localhost', user='root', password='123456', database='lol', charset='utf8')
    cursor = db.cursor()
    sql = "delete from team where team_name = %s "
    val = (team_name,)
    try:
        cursor.execute('START TRANSACTION')
        cursor.execute(sql, val)
        print(f'已删除队伍：{team_name}')
        db.commit()
    except Exception as e:
        db.rollback()  # 若出现错误，则回滚到上一次提交的状态
        print("该队伍曾获得某赛事冠军，无法删除该队伍！！！")

    # 关闭游标和数据库连接
    cursor.close()
    db.close()
#删除异常赛事
def delete_wrong_com():
    db = pymysql.connect(host='localhost', user='root', password='123456', database='lol', charset='utf8')
    cursor = db.cursor()
    sql1 = "delete from team where team_name in (select champion from competition where mvp in (select player_id from player where position is null))"
    sql2 = "delete from competition where mvp in (select player_id from player where position is null)"
    sql3 = "delete from player where position is null"
    try:
        cursor.execute('START TRANSACTION')
        cursor.execute(sql1)
        cursor.execute(sql2)
        cursor.execute(sql3)
        print(f'已删除异常消息')
        db.commit()
    except Exception as e:
        db.rollback()  # 若出现错误，则回滚到上一次提交的状态
        print("删除失败")

    # 关闭游标和数据库连接
    cursor.close()
    db.close()

#更改队名
def update_team_name(old_team_name, new_team_name):
    db = None
    cursor = None
    try:
        db = pymysql.connect(host='localhost', user='root', password='123456', database='lol', charset='utf8')
        cursor = db.cursor()
        sql = "update team set team_name = %s where team_name = %s"
        val = (new_team_name, old_team_name)
        cursor.execute('START TRANSACTION')
        cursor.execute(sql, val)
        print(f"已将队伍{old_team_name}更名为{new_team_name}")
        db.commit()
    except:
        db.rollback()
        print("更改失败")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

#更改队伍选手
def update_team_player(team_name, player_id, position):
    db = None
    cursor = None
    try:
        db = pymysql.connect(host='localhost', user='root', password='123456', database='lol', charset='utf8')
        cursor = db.cursor()
        if position == 'top':
            sql = "update team set top = %s where team_name = %s"
            val = (player_id, team_name)
        elif position == 'jug':
            sql = "update team set jug = %s where team_name = %s"
            val = (player_id, team_name)
        elif position == 'mid':
            sql = "update team set mid = %s where team_name = %s"
            val = (player_id, team_name)
        elif position == 'adc':
            sql = "update team set adc = %s where team_name = %s"
            val = (player_id, team_name)
        elif position == 'sup':
            sql = "update team set sup = %s where team_name = %s"
            val = (player_id, team_name)

        cursor.execute('START TRANSACTION')
        cursor.execute(sql, val)
        print(f'已将队伍{team_name}的{position}位置轮换为{player_id}选手')
        db.commit()
    except:
        db.rollback()
        print("更改失败")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

def update_player_name(old_id, new_id):
    db = None
    cursor = None
    try:
        db = pymysql.connect(host='localhost', user='root', password='123456', database='lol', charset='utf8')
        cursor = db.cursor()
        # sql = "call change_player_name(%s, %s) "
        val = (old_id, new_id)
        # cursor.execute(sql, val)
        cursor.callproc('change_player_name', val)
        db.commit()
    except:
        db.rollback()
        print("更改失败")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

# 打印选手信息
def display_player(text, player_id):  # 建议将参数名改为player_id以符合实际查询条件
    db = None
    cursor = None
    try:
        db = pymysql.connect(
            host='localhost',
            user='root',
            password='123456',
            database='lol',
            charset='utf8'
        )
        cursor = db.cursor()

        # 执行参数化查询
        sql = "SELECT * FROM player WHERE player_id = %s"
        cursor.execute(sql, (player_id,))

        # 获取并显示列名
        columns = [col[0] for col in cursor.description]
        header = " | ".join(f"{col:<15}" for col in columns)  # 使用固定宽度格式化
        text.insert(END, header + "\n")
        text.insert(END, "-" * 20 + "\n")  # 添加分隔线

        # 获取并显示数据
        results = cursor.fetchall()
        for row in results:
            formatted_row = " | ".join(f"{str(field):<15}" for field in row)
            text.insert(END, formatted_row + "\n")

    except pymysql.Error as e:
        text.insert(END, f"数据库错误: {str(e)}\n")
    except Exception as e:
        text.insert(END, f"发生错误: {str(e)}\n")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

# 打印所有选手信息
def display_all_player(text):  # 建议将参数名改为player_id以符合实际查询条件
    db = None
    cursor = None
    try:
        db = pymysql.connect(
            host='localhost',
            user='root',
            password='123456',
            database='lol',
            charset='utf8'
        )
        cursor = db.cursor()

        # 执行参数化查询
        sql = "SELECT * FROM player"
        cursor.execute(sql)

        # 获取并显示列名
        columns = [col[0] for col in cursor.description]
        header = " | ".join(f"{col:<15}" for col in columns)  # 使用固定宽度格式化
        text.insert(END, header + "\n")
        text.insert(END, "-" * 20 + "\n")  # 添加分隔线

        # 获取并显示数据
        results = cursor.fetchall()
        for row in results:
            formatted_row = " | ".join(f"{str(field):<15}" for field in row)
            text.insert(END, formatted_row + "\n")

    except pymysql.Error as e:
        text.insert(END, f"数据库错误: {str(e)}\n")
    except Exception as e:
        text.insert(END, f"发生错误: {str(e)}\n")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()
# 显示某支队伍信息
def display_team(text, team_name):
    try:
        db = pymysql.connect(
            host='localhost',
            user='root',
            password='123456',
            database='lol',
            charset='utf8'
        )
        cursor = db.cursor()

        sql = "SELECT team_name, top, jug, mid, adc, sup FROM team WHERE team_name = %s"
        cursor.execute(sql, (team_name,))
        results = cursor.fetchall()

        if not results:
            text.insert(END, "未找到该战队信息\n")
            return

        # 获取列名
        col_names = [desc[0] for desc in cursor.description]
        text.insert(END, ' | '.join(col_names) + '\n')
        text.insert(END, '-' * 50 + '\n')

        for row in results:
            row_str = ' | '.join(str(item) if item is not None else 'N/A' for item in row)
            text.insert(END, row_str + '\n')

    except pymysql.Error as e:
        text.insert(END, f"数据库错误: {e}\n")
    except Exception as e:
        text.insert(END, f"发生错误: {e}\n")
    finally:
        cursor.close()
        db.close()

# 显示全部队伍的信息
def display_team_all(text):
    db = None
    cursor = None
    try:
        db = pymysql.connect(
            host='localhost',
            user='root',
            password='123456',
            database='lol',  # 注意保持数据库名称正确
            charset='utf8'
        )
        cursor = db.cursor()

        # 执行全表查询
        sql = """SELECT team_name, top, jug, mid, adc, sup 
                FROM team"""
        cursor.execute(sql)

        # 获取并显示列名
        columns = [col[0].upper() for col in cursor.description]  # 列名转大写
        header = " | ".join(f"{col:<12}" for col in columns)
        text.insert(END, header + "\n")
        text.insert(END, "-" * 80 + "\n")  # 根据总宽度调整分隔线

        # 获取并显示数据
        results = cursor.fetchall()
        if not results:
            text.insert(END, "当前没有战队信息\n")
            return

        for row in results:
            # 处理空值并限制字段长度
            formatted_fields = []
            for field in row:
                if not field:
                    formatted_fields.append("N/A".ljust(12))
                else:
                    formatted_fields.append(f"{str(field)[:10]:<12}")  # 限制最大长度
            text.insert(END, " | ".join(formatted_fields) + "\n")

    except pymysql.Error as e:
        text.insert(END, f"数据库错误: {str(e)}\n")
    except Exception as e:
        text.insert(END, f"发生错误: {str(e)}\n")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

# 显示获得某届赛事的冠军队伍及其FMVP选手
def display_competition_info(text, com_name):
    db = None
    cursor = None
    try:
        db = pymysql.connect(
            host='localhost',
            user='root',
            password='123456',
            database='lol',
            charset='utf8'
        )
        cursor = db.cursor()

        # 执行参数化查询
        sql = """SELECT champion, mvp 
                FROM competition 
                WHERE com_name = %s"""
        cursor.execute(sql, (com_name,))

        # 获取并显示列名
        columns = [desc[0].upper() for desc in cursor.description]
        header = " | ".join(f"{col:<20}" for col in columns)
        text.insert(END, f"{"com_name":<20}| {header}\n")
        text.insert(END, "-" * 40 + "\n")

        # 获取并显示数据
        results = cursor.fetchall()
        if not results:
            text.insert(END, f"未找到 {com_name} 的赛事记录\n")
            return

        for row in results:
            formatted_row = f"{com_name:<20} | " + " | ".join(f"{str(field):<20}" for field in row)
            text.insert(END, formatted_row + "\n")

    except pymysql.Error as e:
        text.insert(END, f"数据库错误: {str(e)}\n")
    except Exception as e:
        text.insert(END, f"发生错误: {str(e)}\n")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

# 显示所有的赛事中获得冠军的队伍及其FMVP选手
def display_competition_all(text):
    db = None
    cursor = None
    try:
        db = pymysql.connect(
            host='localhost',
            user='root',
            password='123456',
            database='lol',
            charset='utf8'
        )
        cursor = db.cursor()

        # 执行全表查询
        sql = """SELECT com_name, champion, mvp 
                FROM competition"""
        cursor.execute(sql)

        # 获取并显示列名
        columns = [col[0].upper() for col in cursor.description]  # 转大写
        header = " | ".join(f"{col:<20}" for col in columns)
        text.insert(END, f"{header}\n")
        text.insert(END, "-" * 40 + "\n")

        # 获取并显示数据
        results = cursor.fetchall()
        if not results:
            text.insert(END, "当前没有赛事记录\n")
            return

        for row in results:
            com_name, champion, mvp = row
            formatted_line = (
                f"{com_name:<20} | "
                f"{champion:<20} | "
                f"{mvp:<20}"
            )
            text.insert(END, formatted_line + "\n")

    except pymysql.Error as e:
        text.insert(END, f"数据库错误: {str(e)}\n")
    except Exception as e:
        text.insert(END, f"发生错误: {str(e)}\n")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

#打印流量明星选手
def display_star_players(text):
    db = None
    cursor = None
    try:
        db = pymysql.connect(
            host='localhost',
            user='root',
            password='123456',
            database='lol',  # 根据实际数据库名称修改
            charset='utf8'
        )
        cursor = db.cursor()

        # 查询明星选手视图
        sql = """SELECT * FROM star_player"""
        cursor.execute(sql)

        # 获取并显示列名
        columns = [col[0].upper() for col in cursor.description]  # 列名转大写
        header = " | ".join(f"{col:<15}" for col in columns)
        text.insert(END, header + "\n")
        text.insert(END, "-" * (15 * len(columns) + 3 * (len(columns) - 1)) + "\n")

        # 获取并显示数据
        results = cursor.fetchall()
        if not results:
            text.insert(END, "当前没有明星选手数据\n")
            return

        for row in results:
            # 处理空值和格式化
            formatted_row = " | ".join(
                f"{str(field)[:12]:<15}" if field else "N/A".ljust(15)
                for field in row
            )
            text.insert(END, formatted_row + "\n")

    except pymysql.Error as e:
        text.insert(END, f"数据库错误: {str(e)}\n")
    except Exception as e:
        text.insert(END, f"发生错误: {str(e)}\n")
    finally:
        if cursor:
            cursor.close()
        if db:
            db.close()

