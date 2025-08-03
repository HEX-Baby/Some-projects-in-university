import datetime
import json
import logging
from typing import Dict, List, Optional, Any, TypedDict, Sequence
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from langchain.chains import LLMChain
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph, END, add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing_extensions import Annotated
from models.request_model import TravelPlanRequest
from models.trip_plan_model import TravelPlanResponse, Activity, DayPlan
from tools import ALL_TOOLS  # 所有工具集合
from apis.weather import QWeatherAPI  # 导入天气API类
from datetime import datetime, timedelta  # 用于日期计算

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # 加入 test_planner.py 所在目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))  # 加入项目根目录
# 配置日志
logger = logging.getLogger(__name__)


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    request: Optional[TravelPlanRequest]


class QwenTravelPlanner:
    """基于 Qwen-Max 模型的行程规划器（使用 LangGraph 实现消息持久化）"""

    # 1.创建agent调用工具进行输入输出
    # 2.创建结构化输出，用于提取，辅助agent
    def __init__(self, user: str, session: str):
        self.user = user
        self.session = session
        self.reset()

    def reset(self):
        # 初始化 Qwen-Max 模型
        self.chat_llm = ChatTongyi(model_name='qwen-max', streaming=True)

        # 创建检查点存储器
        self.checkpointer = MemorySaver()

        # 对用户输入的文本进行提取，然后结构化输出
        self.struct_request_llm = self.chat_llm.with_structured_output(TravelPlanRequest)

        # 使用agent根据API搜索的消息进行整合
        self.agent = create_react_agent(self.chat_llm, ALL_TOOLS, checkpointer=self.checkpointer)

        # 对大模型响应的内容进行结构化
        self.struct_response_llm = self.chat_llm.with_structured_output(TravelPlanResponse)

        # 定义工作流的应用程序
        self.app = self._build_graph()

        # 定义会话线程id
        self.config = {'configurable': {'thread_id': self.user + '_' + self.session}}

    def _build_system_prompt(self) -> str:
        """构建详细的行程规划系统提示词，并注入当前日期信息"""
        now = datetime.now()
        current_date_info = f"今天是 {now.strftime('%Y年%m月%d日')}, {['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'][now.weekday()]}。"
        
        return f"""
        你是一名专业的行程规划师，请根据用户需求规划3套不同风格的旅行行程方案。每套方案需包含以下详细信息：

        # 重要时间提示
        - {current_date_info}
        - 当用户提到相对日期时（例如：“明天”、“下周三”、“这个周末”），你必须基于今天的日期进行精确计算，并转换为YYYY-MM-DD格式。

        # 方案要求
        1. 每种方案应有鲜明的主题特色（如：文化探索/美食之旅/自然休闲/亲子欢乐等）
        2. 每个方案必须包含：
           - 清晰的每日时间安排（精确到小时）
           - 每个活动的详细地址和交通方式
           - 餐饮推荐（包含餐厅评分和人均价格）
           - 住宿推荐（包含酒店评分和价格区间）
           - 天气适应性建议
           - 费用预估明细
        3. 活动安排应考虑：
           - 合理的交通时间和景点开放时间
           - 天气状况对户外活动的影响
           - 不同年龄人群的需求

        # 规划原则
        1. 每个方案必须完整独立，不能简单修改
        2. 活动安排要符合当地实际情况
        3. 推荐内容需基于真实数据和用户偏好
        4. 提供专业实用的旅行建议

        # 输出格式要求
        请严格按照以下Markdown格式输出，包含所有指定部分：

        # [方案X名称]之旅
        ## 1. 方案特色
        [简要说明本方案的特色和适合人群]

        ## 2. 每日详细安排
        ### 第N天：YYYY-MM-DD
        - **上午/中午/下午/晚上**（每个时段都需包含）
          - **精确时间**：活动描述
          - **地址**：详细地址
          - **交通**：具体交通方式
          - **备注**：任何注意事项或建议

        ## 3. 餐饮推荐
        - 早餐/午餐/晚餐：
          - 餐厅名称（评分X/X，人均XX元）
          - 推荐菜品
          - 地址和交通

        ## 4. 住宿推荐
        - 酒店名称（评分X/X）
        - 地址
        - 价格区间
        - 特色描述

        ## 5. 天气信息
        - 每日天气状况和具体的温度和穿衣建议

        ## 6. 费用预估
        - 分类明细（餐饮/门票/交通/住宿）
        - 总预算范围

        ## 7. 注意事项
        - 安全提醒
        - 特殊提示
        - 预订建议


        输出的内容一定要准确
        """

    def _build_user_prompt(self, request: TravelPlanRequest) -> str:
        """构建用户提示词"""
        return f"""
        ### 用户旅行需求
        - 目的地: {request.destination}
        - 出行时间: {request.start_date} 至 {request.end_date}（共 {request.duration} 天）
        - 出行人数: {request.travelers} 人（{self._parse_age_groups(request)}）
        - 旅行类型: {request.trip_type}
        - 兴趣偏好: {request.interests or "无特别偏好"}
        - 预算范围: {request.budget or "无限制"}
        - 特殊要求: {request.special_requests or "无"}

        ### 规划要求
        请提供3套不同风格的{request.duration}天行程方案，要求：
        1. 每套方案必须有鲜明主题和特色
        2. 包含所有必要的实用信息（地址、交通、价格等）
        3. 考虑天气因素调整户外活动
        4. 提供专业贴心的旅行建议
        """ + self._build_weather_prompt(request)

    def _parse_age_groups(self, request: TravelPlanRequest) -> str:
        """解析年龄构成"""
        # 这里可以添加从request中解析年龄构成的逻辑
        return "年龄构成未指定"

    def _build_weather_prompt(self, request: TravelPlanRequest) -> str:
        """构建详细的天气提示词"""
        weather_api = QWeatherAPI()
        try:
            import time
            time.sleep(1)  # 增加1秒延迟，避免QPS超限
            weather_data = weather_api.get_city_weather_summary(request.destination)

            if "error" in weather_data:
                return "\n\n⚠️ 天气服务暂时不可用，请根据季节准备常规衣物"

            # 计算行程天数
            try:
                start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
                end_date = datetime.strptime(request.end_date, "%Y-%m-%d")
                days = (end_date - start_date).days + 1
            except:
                days = min(7, request.duration or 7)

            prompt_lines = [
                "\n\n### 目的地天气综合分析",
                "以下天气预报数据可用于调整行程安排："
            ]

            # 当前天气
            if weather_data.get("current"):
                current = weather_data["current"]
                prompt_lines.extend([
                    f"- 当前天气: {current['weather']}，气温 {current['temperature']}°C",
                    f"- 风力: {current['wind_dir']}风 {current['wind_speed']}",
                    f"- 湿度: {current.get('humidity', 'N/A')}%"
                ])

            # 天气预报
            if weather_data.get("daily_forecast"):
                prompt_lines.append("\n### 行程期间每日天气预报")
                for idx, forecast in enumerate(weather_data["daily_forecast"][:days]):
                    day_num = idx + 1
                    date = forecast['datetime']
                    temp = forecast['temperature']
                    weather = forecast['weather']

                    # 生成穿衣建议
                    if "~" in temp:
                        max_temp = int(temp.split("~")[1])
                    else:
                        max_temp = int(temp)

                    clothing = self._get_clothing_suggestion(max_temp, weather)

                    prompt_lines.append(
                        f"**第{day_num}天（{date}）**: "
                        f"{weather}，气温 {temp} | "
                        f"建议穿搭: {clothing}"
                    )

            # 特别提醒
            prompt_lines.append("\n### 天气特别提醒")
            if weather_data.get("daily_forecast"):
                has_rain = any("雨" in f["weather"] for f in weather_data["daily_forecast"][:days])
                has_extreme = any(
                    "雷" in f["weather"] or "暴" in f["weather"] for f in weather_data["daily_forecast"][:days])

                if has_rain:
                    prompt_lines.append("- 部分日期有降雨，建议：")
                    prompt_lines.append("  ✓ 携带折叠伞/雨衣")
                    prompt_lines.append("  ✓ 为户外活动准备备用方案")
                    prompt_lines.append("  ✓ 选择防滑舒适的鞋子")

                if has_extreme:
                    prompt_lines.append("- 部分日期有恶劣天气，建议：")
                    prompt_lines.append("  ✓ 关注当地天气预警")
                    prompt_lines.append("  ✓ 调整户外活动时间")
                    prompt_lines.append("  ✓ 准备应急物品")

            return "\n".join(prompt_lines)
        except Exception as e:
            # 检查是否是高德API的QPS超限错误
            if "CUQPS_HAS_EXCEEDED_THE_LIMIT" in str(e):
                logger.error(f"天气API调用超限: 请稍后再试（{str(e)}）")
                return "\n\n⚠️ 天气查询过于频繁，请1分钟后重试"
            else:
                logger.error(f"天气信息获取失败: {str(e)}")
                return "\n\n⚠️ 天气服务暂时不可用，请根据季节准备常规衣物"

    def _get_clothing_suggestion(self, max_temp: int, weather: str) -> str:
        """根据温度生成穿衣建议"""
        if max_temp > 30:
            return "轻薄夏装、防晒衣、帽子、太阳镜"
        elif max_temp > 25:
            return "短袖+薄外套、舒适便鞋"
        elif max_temp > 15:
            return "长袖衣物、轻便外套"
        elif max_temp > 5:
            return "毛衣/卫衣、防风外套"
        else:
            return "羽绒服、保暖内衣、围巾手套"

    def _get_date_context_prompt(self) -> str:
        """为日期提取生成一个带有当前日期上下文的系统提示。"""
        now = datetime.now()
        # Correctly map weekday() which is 0-6 for Mon-Sun
        weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
        current_date_info = f"今天是 {now.strftime('%Y年%m月%d日')}, {weekdays[now.weekday()]}。"
        return f"""
        你是一个专门负责从用户输入中提取旅行日期的智能助手。
        
        # 必要信息
        - {current_date_info}

        # 你的任务
        - 你必须根据上面的“必要信息”，将用户输入中提到的任何相对日期（例如：“明天”、“后天”、“这个星期五”、“下周一”）准确地解析为“YYYY-MM-DD”格式。
        - 如果用户没有明确或相对地提及日期，请将日期字段留空。
        
        请严格按照此规则处理用户的输入，以提取准确的日期。
        """

    def _build_structured_prompt(self, plan_text: str) -> str:
        return f"""
           请将以下行程计划转换为结构化JSON格式：

           ### 原始行程计划
           {plan_text}

           ### 结构化要求
           请转换为以下JSON格式：
           {{
             "destination": "目的地",
             "start_date": "开始日期",
             "end_date": "结束日期",
             "summary": "行程概述",
             "days": [
               {{
                 "date": "YYYY-MM-DD",
                 "weather": "天气信息",
                 "activities": [
                   {{
                     "time": "时间段",
                     "name": "活动名称",
                     "location": "地点名称",
                     "type": "活动类型",
                     "description": "活动描述"
                   }}
                 ],
                 "hotel": {{"name": "酒店名称", "address": "酒店地址"}}
               }}
             ],
             "notes": ["建议1", "建议2"]
           }}

           请只输出JSON格式的内容，不要包含任何其他文本。
           """

    def handle_request(self, state: State):
        """将用户消息结构化为TravelPlanRequest，并添加到状态中"""
        # 只处理最后一条消息作为用户输入
        last_message = state['messages'][-1].content if state['messages'] else ""

        # 构建包含日期上下文的提示，以帮助模型准确提取日期
        date_context_prompt = self._get_date_context_prompt()
        prompt_with_context = [
            SystemMessage(content=date_context_prompt),
            HumanMessage(content=last_message)
        ]

        # 使用包含上下文的提示来调用结构化模型
        request = self.struct_request_llm.invoke(prompt_with_context, config=self.config)


        # 返回包含请求的状态更新（不添加额外消息）
        return {
            'request': request,
            'messages': []  # 这里返回空列表，不会添加无效消息类型
        }

    def call_agent(self, state: State):
        """处理请求并使用agent生成响应"""
        request = state.get('request')
        if not request:
            logger.error("调用代理时请求对象不存在")
            return {"messages": [AIMessage(content="无法处理请求，缺少必要参数")]}

        try:
            # 1. 构建系统提示和用户提示
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(request)

            # 2. 从状态中获取历史消息
            history = state.get("messages", [])
            history_without_system_and_last = [
                msg for msg in history[:-1] if not isinstance(msg, SystemMessage)
            ]

            # 3. 构造最终消息列表（采纳用户建议，将system_prompt作为HumanMessage）
            # 这可能有助于简化ReAct agent的推理过程
            final_messages_for_agent = [
                HumanMessage(content=system_prompt),
                *history_without_system_and_last,
                HumanMessage(content=user_prompt)
            ]

            # 4. 准备配置，恢复历史记录功能，并增加递归深度限制作为保险
            agent_config = self.config.copy()
            agent_config['recursion_limit'] = 50

            # 5. 调用 agent, 必须传入 config 以保证历史记录和递归限制生效
            agent_output = self.agent.invoke(
                {'messages': final_messages_for_agent},
                config=agent_config
            )

            # 6. 处理输出
            if isinstance(agent_output, dict) and 'messages' in agent_output and agent_output['messages']:
                response_message = agent_output['messages'][-1]
            elif isinstance(agent_output, BaseMessage):
                response_message = agent_output
            else:
                response_message = AIMessage(content=str(agent_output))

            return {
                "messages": [response_message],
                "request": request
            }

        except Exception as e:
            logger.error(f"Agent调用失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "messages": [AIMessage(content=f"行程规划出错: {str(e)}")],
                "request": request
            }

    def _build_graph(self):
        workflow = StateGraph(state_schema=State)

        workflow.add_edge(START, 'struct_request_llm')
        workflow.add_edge('struct_request_llm', 'agent')
        workflow.add_edge('agent', END)

        workflow.add_node('struct_request_llm', self.handle_request)
        workflow.add_node('agent', self.call_agent)

        app = workflow.compile(checkpointer=self.checkpointer)
        return app

    def generate_plan(self, user_input: str):
        """生成旅行计划（流式输出，模仿指定代码风格）"""
        try:
            # 准备初始状态
            initial_state = {"messages": [HumanMessage(content=user_input)], "request": None}

            # 流式调用工作流，指定stream_mode="messages"
            # 迭代获取每个消息片段和元数据
            for chunk, metadata in self.app.stream(
                    initial_state,
                    self.config,  # 使用已定义的会话配置
                    stream_mode="messages"  # 关键：按消息片段流式输出
            ):
                # 过滤并处理模型生成的AIMessage
                if isinstance(chunk, AIMessage):
                    # 实时输出当前token（不换行，强制刷新）
                    print(chunk.content, end="", flush=True)

            print()  # 所有片段输出完毕后换行
            # 返回最后一条完整消息（供历史记录使用）
            return AIMessage(content=chunk.content)  # chunk此时为最后一个片段

        except Exception as e:
            logger.error(f"行程规划失败: {str(e)}")
            print("\n助手: 抱歉，行程规划失败，请重试")
            raise

    def generate_plan_stream(self, user_input: str):
        """流式生成旅行计划，每次yield一段内容"""
        try:
            initial_state = {"messages": [HumanMessage(content=user_input)], "request": None}
            for chunk, metadata in self.app.stream(
                    initial_state,
                    self.config,
                    stream_mode="messages"
            ):
                if isinstance(chunk, AIMessage):
                    yield chunk.content
        except Exception as e:
            yield f"抱歉，行程规划失败: {str(e)}"

    def generate_struct_plan(self, plan_text: str):
        """生成结构化json消息"""
        try:
            # 确保输入是字符串
            if not isinstance(plan_text, str):
                plan_text = str(plan_text)

            prompt = self._build_structured_prompt(plan_text)
            response = self.struct_response_llm.invoke(
                [HumanMessage(content=prompt)],  # 改为直接传入消息列表
                config=self.config
            )
            return response.model_dump_json()
        except Exception as e:
            logger.exception(f"结构化输出失败: {str(e)}")
            return self._fallback_parse_plan(plan_text)  # 确保返回回退结果

    def _fallback_parse_plan(self, plan_text: str) -> TravelPlanResponse:
        """回退解析方法（当结构化解析失败时使用）"""
        # 尝试从文本中提取JSON
        try:
            json_start = plan_text.find('{')
            json_end = plan_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = plan_text[json_start:json_end]
                return TravelPlanResponse.model_validate(json_str)
        except Exception:
            pass

        # 如果无法解析JSON，创建默认响应
        return TravelPlanResponse(
            destination="目的地",
            start_date="开始日期",
            end_date="结束日期",
            days=[]
        )

    def get_conversation_history(self) -> List[Dict]:
        """获取会话历史"""
        try:
            # 从检查点获取状态
            checkpoint = self.checkpointer.get(self.config)
            if checkpoint:
                messages = checkpoint.get('messages', [])
                return [
                    {"role": msg.type, "content": msg.content}
                    for msg in messages
                ]
            return []
        except Exception as e:
            logger.error(f"获取消息历史失败: {str(e)}")
            return []

    def clear_conversation_history(self):
        """清除当前用户的会话历史"""
        try:
            self.checkpointer.delete_thread(self.config['configurable']['thread_id'])
            self.reset()
        except Exception as e:
            logger.error(f"清除会话历史失败: {str(e)}")
#
# ```
# </rewritten_file>