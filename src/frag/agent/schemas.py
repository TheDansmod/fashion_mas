"""Pydantic schemas that constrain the agent's structured LLM outputs."""

from typing import Literal

from pydantic import BaseModel, Field


class ItemToFind(BaseModel):
    """One product to search the catalogue for."""

    item_type: str = Field(
        description=(
            "The kind of product, copied from the item types the user asked for when "
            "they named any (e.g. 'shoes'); otherwise a short name such as 'dress'."
        ),
    )
    description: str = Field(
        description="Catalogue-style description of exactly one product.",
    )


class RequiredClothes(BaseModel):
    """The products to search for."""

    items: list[ItemToFind] = Field(
        min_length=1,
        description="One entry per recommended product; together they satisfy the "
        "user's request.",
    )


class NumRecommendations(BaseModel):
    """How many recommendations the user asked for, and of which item types."""

    num_recommendations: int = Field(
        ge=0,
        le=10,
        description=(
            "Number of recommendations the user explicitly asked for, or 0 if they "
            "did not state a number."
        ),
    )
    item_types: list[str] = Field(
        default_factory=list,
        description=(
            "The kinds of products the user explicitly names, in their own words, "
            "e.g. ['pants', 'shoes']. Empty if they name none (e.g. 'something "
            "similar to this', 'a complete outfit')."
        ),
    )


class SingleImageDescription(BaseModel):
    """Description of one input image."""

    image_description: str = Field(
        min_length=20,
        description="Description of the image, following the instructions.",
    )


class MatchedImageId(BaseModel):
    """The id of the image which was matched to the input description."""

    image_id: int = Field(
        ...,
        description=(
            "Id of the best-matching product; must be an id semantic_search returned."
        ),
    )


class CriticalEvaluation(BaseModel):
    """Whether the recommendations satisfy the request, and how to fix them if not."""

    satisfactory: Literal["Yes", "No"] = Field(
        ...,
        description=(
            "Yes if the recommendations satisfy the user's request, No if they "
            "clearly fail it."
        ),
    )
    correction: str = Field(
        "",
        description=(
            "If No: at most 3 sentences on what to search for instead, restating "
            "every item type the request needs. Empty if Yes."
        ),
    )


class UpdatedUserRequest(BaseModel):
    """A follow-up message rewritten as a standalone request."""

    relevant_image_indexes: list[int] = Field(
        min_length=0,
        description=(
            "Indexes of the earlier images the new request refers to or depends on; "
            "empty if none."
        ),
    )
    updated_user_query: str = Field(
        ...,
        description=(
            "The new message rewritten as a complete standalone request, keeping "
            "the original request's constraints unless the new message changes them."
        ),
    )
