import logging
import re
from typing import List, Dict, Optional
from enum import Enum, auto

# 导入新的依赖
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage

from models.trip_plan_model import TravelPlanResponse
from planner.planner import QwenTravelPlanner
from models.request_model import TravelPlanRequest
from planner.chat_planner import QwenChatPlanner
from models.chat_model import ChatMessage

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TravelAssistant")


class Intent(Enum):
    """用户意图枚举"""
    PLAN_TRIP = auto()
    INQUIRE_HISTORY = auto()
    CHAT = auto()
    UNKNOWN = auto()


class TravelAssistant:
    """整合行程规划与聊天功能的助手（AI意图识别版）"""

    def __init__(self, user: str = "default_user", session: str = "default_session"):
        self.user = user
        self.session = session
        self.trip_planner = QwenTravelPlanner(user=user, session=session)
        self.chat_planner = QwenChatPlanner(user=user, session=session)
        self.logger = logger
        self.conversation_history: List[Dict[str, str]] = []  # 统一管理对话历史

    def _get_user_intent(self, user_input: str) -> Intent:
        """使用LLM判断用户意图"""
        try:
            # 使用轻量级模型进行意图分类，以提高速度和降低成本
            intent_llm = ChatTongyi(model_name="qwen-turbo")
            
            # 为了更准确地判断，只使用最近的几轮对话
            recent_history = self.conversation_history[-3:]
            formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])

            system_prompt = f"""
            你是一个意图分类机器人。根据用户的最新输入和对话历史，判断用户的真实意图。
            可用意图分类:
            1. PLAN_TRIP: 用户明确想要计划或讨论旅行行程、景点、美食、攻略等。例如："我想去北京玩三天", "广州有什么好吃的", "帮我规划一下行程"。
            2. INQUIRE_HISTORY: 用户在询问关于之前对话中提到的信息。例如："我刚刚说要去哪？", "我们上次聊到哪了？", "你还记得我的预算吗？"。
            3. CHAT: 普通闲聊、打招呼、或者不属于以上任何一种的对话。例如："你好", "今天天气怎么样", "你是个机器人吗？"。

            对话历史:
            {formatted_history}

            用户的最新输入: "{user_input}"

            请只返回意图分类的关键词（PLAN_TRIP, INQUIRE_HISTORY, CHAT），并且只能返回一个词。不要返回任何其他内容。
            """

            messages = [SystemMessage(content=system_prompt)]
            response = intent_llm.invoke(messages)
            intent_str = response.content.strip()

            self.logger.info(f"意图识别模型返回: '{intent_str}'")

            if "PLAN_TRIP" in intent_str:
                return Intent.PLAN_TRIP
            elif "INQUIRE_HISTORY" in intent_str:
                return Intent.INQUIRE_HISTORY
            else:
                return Intent.CHAT
        except Exception as e:
            self.logger.error(f"意图识别失败: {e}，默认使用聊天模式。")
            return Intent.CHAT

    def _update_conversation_history(self, role: str, content: str):
        """更新对话历史"""
        self.conversation_history.append({"role": role, "content": content})
        # 保持最近10轮对话，以便提供足够的上下文
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
            
    def _get_full_history_str(self) -> str:
        """获取完整的对话历史字符串"""
        return "\n".join(
            f"{item['role']}: {item['content']}"
            for item in self.conversation_history
        )

    def smart_reply(self, user_input: str) -> str:
        """智能判断并调用对应planner/chat_planner（非流式）"""
        # 1. 更新用户提问到历史记录
        self._update_conversation_history("user", user_input)
        
        # 2. 获取用户意图
        intent = self._get_user_intent(user_input)

        # 3. 根据意图执行
        response_content = ""
        if intent == Intent.PLAN_TRIP:
            self.logger.info("意图: 行程规划. 调用行程规划器...")
            response_content = self.trip_planner.generate_plan(user_input)
        else:  # CHAT, INQUIRE_HISTORY, or UNKNOWN
            self.logger.info(f"意图: {intent.name}. 调用聊天模型...")
            # 同步完整的历史记录到聊天规划器
            self.chat_planner.sync_history(self.conversation_history)
            response_content = self.chat_planner.generate_response(user_input)

        # 4. 更新助手回复到历史记录
        self._update_conversation_history("assistant", response_content)
        return response_content


    def smart_reply_stream(self, user_input: str):
        """流式智能判断并调用对应planner/chat_planner"""
        # 1. 更新用户提问到历史记录，以便意图识别时使用
        self._update_conversation_history("user", user_input)
        
        # 2. 获取意图
        intent = self._get_user_intent(user_input)

        full_response = ""
        response_generator = None

        # 3. 根据意图选择执行器
        if intent == Intent.PLAN_TRIP:
            self.logger.info("意图: 行程规划. 调用行程规划器...")
            response_generator = self.trip_planner.generate_plan_stream(user_input)
        else:  # CHAT, INQUIRE_HISTORY, or UNKNOWN
            self.logger.info(f"意图: {intent.name}. 调用聊天模型...")
            # 同步完整的历史记录到聊天规划器
            self.chat_planner.sync_history(self.conversation_history)
            response_generator = self.chat_planner.generate_response_stream(user_input)

        # 4. 流式返回并收集完整回复
        if response_generator:
            try:
                for chunk in response_generator:
                    full_response += chunk
                    yield chunk
            except Exception as e:
                self.logger.error(f"流式生成失败: {e}")
                fallback_message = "抱歉，我这里出了点问题，稍后再试吧。"
                full_response = fallback_message
                yield fallback_message
        
        # 5. 更新助手完整回复到历史记录
        # 注意：此处更新的是TravelAssistant持有的历史，用于下一次意图判断
        self._update_conversation_history("assistant", full_response)


    def run(self):
        """启动交互循环"""
        self.logger.info("旅行助手已启动 (AI意图识别模式)")
        print("欢迎使用旅行助手！输入'退出'结束对话，输入'清空记忆'重置对话历史\n")
        
        while True:
            try:
                user_input = input("你: ").strip()

                if user_input == "退出":
                    print("助手: 再见！祝你旅途愉快～")
                    break

                if user_input == "清空记忆":
                    self.trip_planner.clear_conversation_history()
                    self.chat_planner.clear_chat_history()
                    self.conversation_history = []
                    print("助手: 已清空对话历史，我们可以重新开始～")
                    continue

                if not user_input:
                    continue

                print("助手: ", end="", flush=True)
                full_response = ""
                for chunk in self.smart_reply_stream(user_input):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print()

            except Exception as e:
                self.logger.error(f"处理请求失败: {str(e)}", exc_info=True)
                print("\n助手: 抱歉，处理时出错了，请再试一次～\n")


if __name__ == "__main__":
    assistant = TravelAssistant(user="test_user", session="test_session_001")
    assistant.run()