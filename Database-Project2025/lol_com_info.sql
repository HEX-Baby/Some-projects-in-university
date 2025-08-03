CREATE DATABASE IF NOT EXISTS lol
DEFAULT CHARACTER SET utf8mb4
DEFAULT COLLATE utf8mb4_unicode_ci;

use lol;

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";

--
-- 表的结构 `team`
--

create table team(
	team_name varchar(255),
    top varchar(255) ,
    jug varchar(255) ,
    mid varchar(255) ,
    adc varchar(255) ,
    sup varchar(255) ,
    primary key(team_name)
);

--
-- 表的结构 `player`
--
create table player(
	player_id varchar(255),
    position varchar(255) ,
    primary key(player_id)
);

--
-- 表的结构 `competition`
--
create table competition(
	com_name varchar(255),
    mvp varchar(255) ,
    champion varchar(255) ,
    primary key(com_name)
);


--
-- 限制表 `team`
--
alter table team
	add constraint top foreign key(top) references player(player_id) on delete set null on update cascade;
alter table team
	add constraint jug foreign key(jug) references player(player_id) on delete set null on update cascade;
alter table team
	add constraint mid foreign key(mid) references player(player_id) on delete set null on update cascade;
alter table team
	add constraint adc foreign key(adc) references player(player_id) on delete set null on update cascade;
alter table team
	add constraint sup foreign key(sup) references player(player_id) on delete set null on update cascade;
    
--
-- 限制表 `competition`
--

alter table competition
	add constraint mvp foreign key(mvp) references player(player_id) on delete set null on update cascade;
alter table competition
	add constraint champion foreign key(champion) references team(team_name) on delete set null on update cascade;

--
-- 转存表中的数据 `player`
--
insert into player values
('Flandre', 'top'),
('SofM', 'jug'),
('Guoguo', 'mid'),
('kRYST4L', 'adc'),
('Hudie', 'sup'),
('Letme', 'top'),
('Mlxg', 'jug'),
('xiaohu', 'mid'),
('Uzi', 'adc'),
('Ming', 'sup'),
('Mouse', 'top'),
('Flawless', 'jug'),
('Doinb', 'mid'),
('SMLZ', 'adc'),
('Killula', 'sup'),
('Zoom', 'top'),
('Clid', 'jug'),
('Yagao', 'mid'),
('LokeN', 'adc'),
('LvMao', 'sup'),
('Gimgoon', 'top'),
('Pepper', 'jug'),
('bing', 'mid'),
('lwx', 'adc'),
('Crisp', 'sup'),
('Theshy', 'top'),
('Ning', 'jug'),
('Rookie', 'mid'),
('JackeyLove', 'adc'),
('Baolan', 'sup'),
('Ray', 'top'),
('Clearlove', 'jug'),
('Scout', 'mid'),
('iBoy', 'adc'),
('Meiko', 'sup'),
('957', 'top'),
('Condi', 'jug'),
('xiye', 'mid'),
('Mystic', 'adc'),
('Ben', 'sup'),
('Xiaoxu', 'top'),
('Xiaohao', 'jug'),
('Vicla', 'mid'),
('Assum', 'adc'),
('Jwei', 'sup'),
('Cuvee', 'top'),
('Ambition', 'jug'),
('Crown', 'mid'),
('Ruler', 'adc'),
('Corejj', 'sup'),
('Thal', 'top'),
('Blank', 'jug'),
('Faker', 'mid'),
('Bang', 'adc'),
('Wolf', 'sup'),
('Mata', 'sup'),
('Karsa', 'jug');
--
-- 转存表中的数据 `team`
--

insert into team values
('SNAKE', 'Flandre', 'SofM', 'Guoguo', 'kRYST4L', 'Hudie'),
('RNG', 'Letme', 'Mlxg', 'xiaohu', 'Uzi', 'Ming'),
('RW', 'Mouse', 'Flawless', 'Doinb', 'SMLZ', 'Killula'),
('JDG', 'Zoom', 'Clid', 'Yagao', 'LokeN', 'LvMao'),
('FPX', 'Gimgoon', 'Pepper', 'bing', 'lwx', 'Crisp'),
('IG', 'Theshy', 'Ning', 'Rookie', 'JackeyLove', 'Baolan'),
('EDG', 'Ray', 'Clearlove', 'Scout', 'iBoy', 'Meiko'),
('WE', '957', 'Condi', 'xiye', 'Mystic', 'Ben'),
('中国队', 'Letme', 'Mlxg', 'xiye', 'Uzi', 'Ming'),
('RA', 'Xiaoxu', 'Xiaohao', 'Vicla', 'Assum', 'Jwei');
--
-- 转存表中的数据 `competition`
--
insert into competition values
('s8', 'Ning', 'IG'),
('2018MSI', 'Uzi', 'RNG'),
('2018LPL春决', 'Uzi', 'RNG'),
('2018LPL夏决', 'Uzi', 'RNG');


--
-- 触发器 `competition`
--
drop trigger if exists competition_tri_update;
delimiter //
create trigger competition_tri_update
before update on competition for each row
begin
if not exists (select * from player where new.mvp = player_id)
then SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'mvp not in player table!';
end if;

if not exists (select * from team where new.champion = team_name)
then update team set team_name = new.champion where team_name = old.champion;
end if;
end; //
delimiter ;

drop trigger if exists competition_tri_insert;
delimiter //
create trigger competition_tri_insert
before insert on competition for each row
begin
if not exists (select * from player where new.mvp = player_id)
then SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'mvp not in player table!';
end if;

if not exists (select * from team where new.champion = team_name)
then SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'team not in team table!';
end if;
end; //
delimiter ;

--
-- 触发器 `team`
--
drop trigger if exists team_top_update_tri;
delimiter //
create trigger team_top_update_tri
before update on team for each row
begin
if new.top not in (select player_id from player where position = 'top')
then SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'the top player does not exist!';
end if;
end;//
delimiter ;

drop trigger if exists team_jug_update_tri;
delimiter //
create trigger team_jug_update_tri
before update on team for each row
begin
if new.jug not in (select player_id from player where position = 'jug')
then SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'the jug player does not exist!';
end if;
end;//
delimiter ;

drop trigger if exists team_mid_update_tri;
delimiter //
create trigger team_mid_update_tri
before update on team for each row
begin
if new.mid not in (select player_id from player where position = 'mid')
then SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'the mid player does not exist!';
end if;
end;//
delimiter ;

drop trigger if exists team_adc_update_tri;
delimiter //
create trigger team_adc_update_tri
before update on team for each row
begin
if new.adc not in (select player_id from player where position = 'adc')
then SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'the adc player does not exist!';
end if;
end;//
delimiter ;


drop trigger if exists team_sup_update_tri;
delimiter //
create trigger team_sup_update_tri
before update on team for each row
begin
if new.sup not in (select player_id from player where position = 'sup')
then SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'the sup player does not exist!';
end if;
end;//
delimiter ;
--
-- 替换视图以便查看 `star player(流量明星选手)`
--
create view star_player as
select *
from player
where player_id in ('Uzi', 'Theshy', 'Faker', 'Clearlove', 'Mlxg');

--
-- 替换视图以便查看 `LPL决赛`
--
create view LPL_final as
select *
from competition
where com_name like '%LPL%决%';

