"""
Skill 基类 - 所有可复用能力单元的基础
参考 Claude Code 插件的 Skill 设计模式
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar, Generic, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SkillStatus(Enum):
    """Skill 执行状态"""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    SKIP = "skip"


T = TypeVar('T')


@dataclass
class SkillResult:
    """
    Skill 执行结果

    所有 Skill 的 execute() 方法都应该返回这个类型

    Attributes:
        success: 是否成功
        status: 详细状态
        data: 结果数据
        confidence: 置信度 (0-1)
        metadata: 额外元数据
        error: 错误信息
        execution_time: 执行时间(毫秒)
    """
    success: bool
    status: SkillStatus = SkillStatus.SUCCESS
    data: Optional[T] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0

    def __post_init__(self):
        if self.status == SkillStatus.FAILURE:
            self.success = False
        elif self.status == SkillStatus.SUCCESS:
            self.success = True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'status': self.status.value,
            'data': self._serialize_data(),
            'confidence': self.confidence,
            'metadata': self.metadata,
            'error': self.error,
            'execution_time': self.execution_time,
        }

    def _serialize_data(self) -> Any:
        """序列化数据"""
        if self.data is None:
            return None
        if isinstance(self.data, (str, int, float, bool, list, dict)):
            return self.data
        if hasattr(self.data, 'to_dict'):
            return self.data.to_dict()
        if hasattr(self.data, '__dict__'):
            return self.data.__dict__
        return str(self.data)


class BaseSkill(ABC, Generic[T]):
    """
    Skill 基类

    所有 Skill 必须实现:
    - execute(): 执行能力，返回 SkillResult
    - validate(): 验证输入参数 (可选)

    设计原则:
    1. 单一职责 - 只做一件事
    2. 无状态 - 不保存调用之间的状态
    3. 可组合 - 可被其他 Skill/Agent 使用
    4. 可测试 - 容易编写单元测试
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 Skill

        Args:
            config: Skill 配置参数
        """
        self.config = config or {}
        self._cache: Dict[str, Any] = {}
        self._execution_count = 0
        self._success_count = 0

    @abstractmethod
    def execute(self, **kwargs) -> SkillResult:
        """
        执行 Skill 的核心能力

        Returns:
            SkillResult: 执行结果
        """
        pass

    def validate(self, **kwargs) -> tuple[bool, Optional[str]]:
        """
        验证输入参数

        Returns:
            (is_valid, error_message)
        """
        return True, None

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """设置配置项"""
        self.config[key] = value

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        success_rate = self._success_count / self._execution_count if self._execution_count > 0 else 0
        return {
            'execution_count': self._execution_count,
            'success_count': self._success_count,
            'success_rate': success_rate,
        }

    def _execute_with_tracking(self, **kwargs) -> SkillResult:
        """内部执行方法，带有统计跟踪"""
        start_time = datetime.now()

        # 验证输入
        is_valid, error_msg = self.validate(**kwargs)
        if not is_valid:
            self._execution_count += 1
            return SkillResult(
                success=False,
                status=SkillStatus.FAILURE,
                error=f"参数验证失败: {error_msg}",
                execution_time=(datetime.now() - start_time).total_seconds() * 1000
            )

        # 执行核心逻辑
        result = self.execute(**kwargs)

        # 更新统计
        self._execution_count += 1
        if result.success:
            self._success_count += 1

        # 设置执行时间
        result.execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return result


class SkillRegistry:
    """
    Skill 注册表
    管理所有可用的 Skill
    """

    _instance: Optional['SkillRegistry'] = None
    _skills: Dict[str, type[BaseSkill]] = {}

    @classmethod
    def get_instance(cls) -> 'SkillRegistry':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def register(cls, name: str, skill_class: type[BaseSkill]) -> None:
        """注册 Skill"""
        cls._skills[name] = skill_class

    @classmethod
    def create(cls, name: str, config: Optional[Dict[str, Any]] = None) -> Optional[BaseSkill]:
        """创建 Skill 实例"""
        skill_class = cls._skills.get(name)
        if skill_class is None:
            return None
        return skill_class(config=config)

    @classmethod
    def list_skills(cls) -> List[str]:
        """列出所有已注册的 Skill"""
        return list(cls._skills.keys())


def register_skill(name: str):
    """
    Skill 注册装饰器

    使用示例:
    @register_skill('fractal')
    class FractalSkill(BaseSkill):
        pass
    """
    def decorator(skill_class: type[BaseSkill]) -> type[BaseSkill]:
        SkillRegistry.register(name, skill_class)
        return skill_class
    return decorator


class SkillChain:
    """
    Skill 链
    按顺序执行多个 Skill，前一个的输出作为后一个的输入
    """

    def __init__(self, skills: List[BaseSkill]):
        """
        初始化 Skill 链

        Args:
            skills: Skill 列表，按执行顺序排列
        """
        self.skills = skills

    def execute(self, initial_kwargs: Dict[str, Any]) -> List[SkillResult]:
        """
        执行 Skill 链

        Args:
            initial_kwargs: 初始参数

        Returns:
            List[SkillResult]: 每个 Skill 的执行结果
        """
        results = []
        current_kwargs = initial_kwargs.copy()

        for skill in self.skills:
            result = skill._execute_with_tracking(**current_kwargs)
            results.append(result)

            # 如果 Skill 失败，停止执行链
            if not result.success:
                break

            # 将当前 Skill 的数据合并到下一个 Skill 的参数中
            if result.data is not None:
                current_kwargs.update(self._data_to_dict(result.data))

        return results

    def _data_to_dict(self, data: Any) -> Dict[str, Any]:
        """将数据转换为字典"""
        if isinstance(data, dict):
            return data
        if hasattr(data, '__dict__'):
            return data.__dict__
        return {'data': data}


class SkillPipeline:
    """
    Skill 管道
    并行执行多个 Skill，然后合并结果
    """

    def __init__(self, skills: List[BaseSkill], merge_strategy: str = 'all'):
        """
        初始化 Skill 管道

        Args:
            skills: Skill 列表
            merge_strategy: 合并策略 ('all'=全部成功, 'any'=任一成功, 'majority'=多数成功)
        """
        self.skills = skills
        self.merge_strategy = merge_strategy

    def execute(self, **kwargs) -> SkillResult:
        """
        执行 Skill 管道

        Args:
            **kwargs: 传递给所有 Skill 的参数

        Returns:
            SkillResult: 合并后的结果
        """
        results = []
        for skill in self.skills:
            result = skill._execute_with_tracking(**kwargs)
            results.append(result)

        return self._merge_results(results)

    def _merge_results(self, results: List[SkillResult]) -> SkillResult:
        """合并多个 Skill 的结果"""
        if not results:
            return SkillResult(success=False, error="没有 Skill 执行结果")

        successful = [r for r in results if r.success]

        if self.merge_strategy == 'all':
            if len(successful) != len(results):
                return SkillResult(
                    success=False,
                    error=f"部分 Skill 失败: {len(successful)}/{len(results)} 成功"
                )

        elif self.merge_strategy == 'any':
            if not successful:
                return SkillResult(success=False, error="所有 Skill 都失败了")

        elif self.merge_strategy == 'majority':
            if len(successful) < len(results) / 2:
                return SkillResult(
                    success=False,
                    error=f"多数 Skill 失败: {len(successful)}/{len(results)} 成功"
                )

        # 合并数据
        merged_data = {}
        merged_metadata = {}
        total_confidence = 0

        for result in successful:
            if result.data is not None:
                if isinstance(result.data, dict):
                    merged_data.update(result.data)
                else:
                    merged_data[result.__class__.__name__] = result.data
            merged_metadata.update(result.metadata)
            total_confidence += result.confidence

        # 平均置信度
        avg_confidence = total_confidence / len(successful) if successful else 0

        return SkillResult(
            success=True,
            data=merged_data,
            confidence=avg_confidence,
            metadata={
                'merged_from': len(successful),
                'total_skills': len(results),
                **merged_metadata
            }
        )
