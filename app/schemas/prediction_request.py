"""Pydantic request schemas for prediction endpoints and message queues."""

from typing import Union
from pydantic import BaseModel, Field, ConfigDict


class CommentItem(BaseModel):
    """Represents a single comment with its database ID if available."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    comment_id: int | None = Field(default=None, alias="commentId")
    comment: str = ""


class ClassMetricInput(BaseModel):
    """Represents a single analyzed class metric vector and its associated comments."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    class_id: int | None = Field(default=None, alias="classId")
    class_name: str = Field(default="Unknown", alias="className")
    file_path: str = Field(default="", alias="filePath")
    start_line: int = Field(default=0, alias="startLine")
    end_line: int = Field(default=0, alias="endLine")
    comments: list[Union[CommentItem, str]] = Field(default_factory=list)

    # 28 Canonical Bug Prediction Features
    cbo: int = 0
    dit: int = 0
    fan_in: int = Field(default=0, alias="fanIn")
    fan_out: int = Field(default=0, alias="fanOut")
    lcom: float = 0.0
    noc: int = 0
    number_of_attributes: int = Field(default=0, alias="numberOfAttributes")
    number_of_lines_of_code: int = Field(default=0, alias="numberOfLinesOfCode")
    number_of_methods: int = Field(default=0, alias="numberOfMethods")
    number_of_private_attributes: int = Field(default=0, alias="numberOfPrivateAttributes")
    number_of_private_methods: int = Field(default=0, alias="numberOfPrivateMethods")
    number_of_public_attributes: int = Field(default=0, alias="numberOfPublicAttributes")
    number_of_public_methods: int = Field(default=0, alias="numberOfPublicMethods")
    rfc: int = 0
    wmc: float = 0.0
    number_of_versions_until: int = Field(default=0, alias="numberOfVersionsUntil")
    number_of_authors_until: int = Field(default=0, alias="numberOfAuthorsUntil")
    lines_added_until: int = Field(default=0, alias="linesAddedUntil")
    max_lines_added_until: int = Field(default=0, alias="maxLinesAddedUntil")
    avg_lines_added_until: float = Field(default=0.0, alias="avgLinesAddedUntil")
    lines_removed_until: int = Field(default=0, alias="linesRemovedUntil")
    max_lines_removed_until: int = Field(default=0, alias="maxLinesRemovedUntil")
    avg_lines_removed_until: float = Field(default=0.0, alias="avgLinesRemovedUntil")
    code_churn_until: int = Field(default=0, alias="codeChurnUntil")
    max_code_churn_until: int = Field(default=0, alias="maxCodeChurnUntil")
    avg_code_churn_until: float = Field(default=0.0, alias="avgCodeChurnUntil")
    age_with_respect_to: float = Field(default=0.0, alias="ageWithRespectTo")
    weighted_age_with_respect_to: float = Field(default=0.0, alias="weightedAgeWithRespectTo")


class PredictionJobRequest(BaseModel):
    """Payload sent when requesting batch predictions for an entire analysis job."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    job_id: str = Field(alias="jobId", default="")
    repository_id: str | None = Field(default=None, alias="repositoryId")
    classes: list[ClassMetricInput] = Field(default_factory=list)


class SATDCommentRequest(BaseModel):
    """Request payload for predicting a single comment's technical debt category."""

    comment_id: int | None = Field(default=None, alias="commentId")
    comment: str