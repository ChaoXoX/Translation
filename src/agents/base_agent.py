"""
基础Agent类

定义所有Agent的通用接口和功能
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pydantic import BaseModel
import json
import re
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from config.settings import get_settings


class AgentState(BaseModel):
    """Agent状态"""
    name: str
    status: str = "idle"  # idle, running, completed, error
    current_task: str = ""
    progress: float = 0.0
    error_message: str = ""
    result: Any = None


class BaseAgent(ABC):
    """基础Agent类"""
    
    def __init__(
        self, 
        name: str, 
        llm: Optional[ChatOpenAI] = None,
        temperature: float = 0.3
    ):
        self.name = name
        self.settings = get_settings()
        
        # 初始化LLM
        if llm:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                api_key=self.settings.llm.api_key,
                base_url=self.settings.llm.base_url,
                model=self.settings.llm.model_name,
                temperature=temperature,
                max_tokens=self.settings.llm.max_tokens
            )
        
        # Agent状态
        self.state = AgentState(name=name)
        
        # 消息历史
        self.message_history: List[Any] = []
        
        # 统计本Agent调用LLM的次数
        self.llm_calls: int = 0
        
        logger.info(f"Agent '{name}' initialized")
    
    @abstractmethod
    async def run(self, input_data: Any) -> Any:
        """执行Agent任务（子类必须实现）"""
        pass
    
    async def invoke_llm(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        parse_json: bool = True,
        max_retries: int = 3
    ) -> Any:
        """调用LLM（带重试）"""
        import asyncio
        last_error = None
        
        for attempt in range(max_retries):
            try:
                messages = []
                
                if system_prompt:
                    messages.append(SystemMessage(content=system_prompt))
                
                messages.append(HumanMessage(content=prompt))
                
                # 调用LLM
                response = await self.llm.ainvoke(messages)
                content = response.content
                
                # 统计调用次数
                self.llm_calls += 1
                break  # 成功则跳出
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.warning(f"Agent '{self.name}' LLM call failed (attempt {attempt+1}/{max_retries}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Agent '{self.name}' LLM call failed after {max_retries} attempts: {e}")
                    raise
        
        try:
            
            # 记录历史
            self.message_history.append({
                "role": "user",
                "content": prompt[:200] + "..." if len(prompt) > 200 else prompt
            })
            self.message_history.append({
                "role": "assistant",
                "content": content[:200] + "..." if len(content) > 200 else content
            })
            
            # 解析JSON
            if parse_json:
                return self._parse_json_response(content)
            
            return content
            
        except Exception as e:
            logger.error(f"Agent '{self.name}' LLM call failed: {e}")
            raise
    
    def _parse_json_response(self, content: str) -> Any:
        """解析JSON响应"""
        try:
            # 尝试直接解析
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        # 尝试提取JSON块
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}',
            r'\[[\s\S]*\]'
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    continue
        
        # 返回原始内容
        logger.warning(f"Failed to parse JSON from response, returning raw content")
        return {"raw_content": content}
    
    def update_state(self, status: str, task: str = "", progress: float = 0.0):
        """更新Agent状态"""
        self.state.status = status
        self.state.current_task = task
        self.state.progress = progress
        logger.debug(f"Agent '{self.name}' state: {status}, task: {task}, progress: {progress:.1%}")
    
    def set_error(self, error_message: str):
        """设置错误状态"""
        self.state.status = "error"
        self.state.error_message = error_message
        logger.error(f"Agent '{self.name}' error: {error_message}")
    
    def get_state(self) -> AgentState:
        """获取Agent状态"""
        return self.state
    
    def reset(self):
        """重置Agent状态"""
        self.state = AgentState(name=self.name)
        self.message_history = []
        logger.info(f"Agent '{self.name}' reset")
