"""Skill records exchanged with optional education-wallet integrations."""

from dataclasses import dataclass
from enum import Enum


class SkillCategory(str, Enum):
    TECHNICAL = "technical"
    COMMUNICATION = "communication"
    SOFT = "soft"


class SkillLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class Skill:
    skill_id: str
    name: str
    description: str
    category: SkillCategory
    level: SkillLevel
