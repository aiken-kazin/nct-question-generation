from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class QuestionOption(BaseModel):
    label: str
    text: str


class FigureSpec(BaseModel):
    figure_type: str
    parameters: dict
    caption: str


class GeneratedQuestion(BaseModel):
    """Raw structured output from the Generator Agent."""
    topic: str
    question_text: str
    options: list[QuestionOption] = Field(min_length=4, max_length=4)
    correct_answer: str  # "A", "B", "C", or "D"
    explanation: str
    latex_formulas: list[str] = Field(default_factory=list)
    figure_spec: Optional[FigureSpec] = None


class DimensionScores(BaseModel):
    correctness: float = Field(ge=0, le=10)
    distractor_quality: float = Field(ge=0, le=10)
    difficulty_alignment: float = Field(ge=0, le=10)
    kazakh_language_quality: float = Field(ge=0, le=10)
    latex_validity: float = Field(ge=0, le=10)
    figure_relevance: Optional[float] = Field(default=None, ge=0, le=10)


class CriticFeedback(BaseModel):
    critic_solution: str
    critic_answer: str
    dimensions: DimensionScores
    overall_score: float = Field(ge=0, le=10)
    pass_fail: bool
    comments: str
    improvement_suggestions: Optional[str] = None


class Question(BaseModel):
    """Final output question with all metadata."""
    id: str
    subject: str
    level: str
    format: str
    topic: str
    question_text: str
    options: list[QuestionOption]
    correct_answer: str
    explanation: str
    latex_formulas: list[str]
    figure_spec: Optional[FigureSpec] = None
    figure_path: Optional[str] = None
    critic_score: Optional[float] = None
    critic_feedback: Optional[CriticFeedback] = None
    timestamp: str
    generation_attempts: int = 1
