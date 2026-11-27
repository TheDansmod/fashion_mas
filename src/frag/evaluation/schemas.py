"""Schema for the LLM-as-a-judge scores."""

from pydantic import BaseModel, Field


class RecommendationEvaluation(BaseModel):
    """Class to evaluate the provided recommendations on various metrics."""

    visual_grounding_score: int = Field(
        ...,
        ge=0,
        le=10,
        description=(
            "An integer score between 1 and 10 checking if the recommendations "
            "demonstrably reflect what is present in the uploaded image(s) if they "
            "are present. If there are no images, give a score of 0 for visual "
            "grounding."
        ),
    )
    visual_grounding_explanation: str = Field(
        ...,
        description=(
            "A single sentence explanation justifying or explaining the score given"
            " for visual grounding."
        ),
    )
    item_suitability_score: int = Field(
        ...,
        ge=1,
        le=10,
        description=(
            "Average integer score for how well each recommended item satisfies its"
            " use case. 1, if most items are unsuitable. 10, if all items are well "
            "justified in their inclusion."
        ),
    )
    item_suitability_explanation: str = Field(
        ...,
        description=(
            "A single sentence explanation justifying or explaining the score given"
            " for item suitability."
        ),
    )
    completeness_coverage_score: int = Field(
        ...,
        ge=1,
        le=10,
        description=(
            "An integer score between 1 and 10 checking if the responses as a whole"
            " address the full scope of the request. 1, if major aspects of the "
            "request are unaddressed. 10, if every element of the request is "
            "covered."
        ),
    )
    completeness_coverage_explanation: str = Field(
        ...,
        description=(
            "A single sentence explanation justifying or explaining the score given"
            " for completeness and coverage."
        ),
    )
