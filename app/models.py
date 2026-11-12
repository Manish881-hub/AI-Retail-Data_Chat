"""The structured query plan. The LLM may *propose* one of these; code validates and executes it.

`extra="forbid"` everywhere: unknown keys from a model are a validation error, not silently ignored.
"""
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

Scalar = Union[str, int, float, bool]
AGGS = ("sum", "mean", "median", "min", "max", "count", "nunique", "aov")
OPS = ("eq", "ne", "gt", "gte", "lt", "lte", "in", "contains")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Filter(_Strict):
    field: str
    op: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "contains"]
    value: Union[Scalar, list[Scalar]]


class Metric(_Strict):
    agg: Literal["sum", "mean", "median", "min", "max", "count", "nunique", "aov"]
    field: str
    as_: str = Field(alias="as")


class Join(_Strict):
    dataset: str
    left_on: list[str]
    right_on: list[str]
    how: Literal["left", "inner"] = "left"


class Sort(_Strict):
    field: str
    dir: Literal["asc", "desc"] = "desc"


class Plan(_Strict):
    intent: Literal["aggregate", "list", "describe", "distinct", "retail_score", "clarify", "unanswerable"]
    dataset: Optional[str] = None
    version: Literal["raw", "clean"] = "clean"
    joins: list[Join] = []
    filters: list[Filter] = []
    group_by: list[str] = []
    metrics: list[Metric] = []
    select: list[str] = []
    sort: list[Sort] = []
    limit: Optional[int] = None
    inherit_context: bool = False
    message: Optional[str] = None

    def dump(self) -> dict:
        return self.model_dump(by_alias=True)
