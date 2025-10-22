"""
策略模块 - 包含用于 Self-Play 的基准策略
"""

from .baseline_policies import RandomPlayerPolicy, MinMaxPlayerPolicy

__all__ = ['RandomPlayerPolicy', 'MinMaxPlayerPolicy']
