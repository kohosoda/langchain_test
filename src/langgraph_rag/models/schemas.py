"""AI応答を構造化して出力させるためのスキーマ定義。"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class QuestionCategory(str, Enum):
    """質問を分類するカテゴリ。"""

    AI_DEVELOPMENT = "ai_dev"
    PRODUCT = "product"
    TECH = "tech"
    OTHER = "other"


class CategoryClassification(BaseModel):
    """質問分類の結果。"""

    category: QuestionCategory = Field(description="質問のカテゴリ")
    confidence: float = Field(
        description="分類の信頼度 (0.0-1.0)", ge=0.0, le=1.0
    )
    reasoning: str = Field(description="分類理由の簡潔な説明")


class RelevanceGrade(BaseModel):
    """ドキュメント関連性の評価結果。"""

    is_relevant: bool = Field(description="ドキュメントが質問に関連しているか")
    confidence: float = Field(
        description="評価の信頼度 (0.0-1.0)", ge=0.0, le=1.0
    )
    reasoning: str = Field(description="評価理由の簡潔な説明")


class GroundingCheck(BaseModel):
    """生成した回答が情報源に基づいているかの検証結果。"""

    is_grounded: bool = Field(description="回答がドキュメントに基づいているか")
    confidence: float = Field(
        description="検証の信頼度 (0.0-1.0)", ge=0.0, le=1.0
    )
    reasoning: str = Field(description="判定理由の簡潔な説明")


class GenerationResult(BaseModel):
    """回答生成処理で返した補助情報。"""

    answer: str = Field(description="生成された回答本文")
    reasoning: Optional[str] = Field(
        default=None, description="回答時に考慮したポイント"
    )
