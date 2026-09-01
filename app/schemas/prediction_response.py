"""Pydantic response schemas for prediction results."""

from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict


class SATDDetectionResponse(BaseModel):
    """Result of classifying a single source comment for SATD."""

    model_config = ConfigDict(populate_by_name=True)

    comment_id: int | None = Field(default=None, alias="commentId")
    comment: str
    category: str
    confidence_score: float = Field(alias="confidenceScore")
    is_debt: bool = Field(alias="isDebt")


class BugPredictionResponse(BaseModel):
    """Result of running bug prediction on a class metric vector."""

    model_config = ConfigDict(populate_by_name=True)

    prediction: int
    probability_score: float = Field(alias="probabilityScore")
    is_defective: bool = Field(alias="isDefective")


class ClassPredictionResponse(BaseModel):
    """Combined bug and SATD prediction output for a single class."""

    model_config = ConfigDict(populate_by_name=True)

    class_id: int | None = Field(default=None, alias="classId")
    class_name: str = Field(alias="className")
    file_path: str = Field(alias="filePath")
    start_line: int = Field(alias="startLine")
    end_line: int = Field(alias="endLine")
    bug_prediction: BugPredictionResponse = Field(alias="bugPrediction")
    satd_detections: list[SATDDetectionResponse] = Field(default_factory=list, alias="satdDetections")


class PredictionJobResponse(BaseModel):
    """Complete batch prediction response returned over HTTP or RabbitMQ."""

    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(alias="jobId")
    repository_id: str | None = Field(default=None, alias="repositoryId")
    status: str = "SUCCESS"
    total_classes_analyzed: int = Field(alias="totalClassesAnalyzed")
    defective_classes_count: int = Field(alias="defectiveClassesCount")
    total_comments_classified: int = Field(alias="totalCommentsClassified")
    total_satd_count: int = Field(alias="totalSatdCount")
    classes: list[ClassPredictionResponse] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())