"""The pydantic schemas used to constrain LLM outputs."""

from typing import Literal

from pydantic import BaseModel, Field


class RequiredClothes(BaseModel):
    """Class to constrain LLM output when generating clothing item descriptions."""

    required_clothes_descriptions: list[str] = Field(
        min_length=1,
        description=(
            "List of descriptions of clothing items that satisfy the user's"
            "requests. One description per index."
        ),
    )


class NumRecommendations(BaseModel):
    """Constrains LLM output when getting the no. of reco's in the user request."""

    num_recommendations: int = Field(
        ge=1,
        le=10,
        description=(
            "Number of recommendations specified by the user, 0 if no"
            "specification made by the user."
        ),
    )


class SingleImageDescription(BaseModel):
    """Constrains LLM output when generating image descriptions."""

    image_description: str = Field(
        min_length=20,
        description="Description of the image as mentioned in the instructions.",
    )


class ValidCategories(BaseModel):
    """List of valid dataset categories to which the clothing item must belong."""

    categories: list[
        Literal[
            "CLUTCHES & POUCHES",
            "POUCHES & DOCUMENT HOLDERS",
            "BOOTS",
            "BACKPACKS",
            "SWEATERS",
            "SWIMWEAR",
            "MONKSTRAPS",
            "JEWELRY",
            "DUFFLE & TOP HANDLE BAGS",
            "JEANS",
            "LACE UPS",
            "SKIRTS",
            "DUFFLE BAGS",
            "TOPS",
            "DRESSES",
            "MESSENGER BAGS & SATCHELS",
            "SOCKS",
            "LOAFERS",
            "ESPADRILLES",
            "UNDERWEAR & LOUNGEWEAR",
            "BAG ACCESSORIES",
            "HATS",
            "SANDALS",
            "JACKETS & COATS",
            "MESSENGER BAGS",
            "GLOVES",
            "TRAVEL BAGS",
            "LINGERIE",
            "SCARVES",
            "KEYCHAINS",
            "BLANKETS",
            "TIES",
            "FLATS",
            "SHORTS",
            "PANTS",
            "SUITS & BLAZERS",
            "TOTE BAGS",
            "HEELS",
            "EYEWEAR",
            "BRIEFCASES",
            "JUMPSUITS",
            "FINE JEWELRY",
            "BELTS & SUSPENDERS",
            "SHIRTS",
            "BOAT SHOES & MOCCASINS",
            "SNEAKERS",
            "POCKET SQUARES & TIE BARS",
            "SHOULDER BAGS",
        ]
    ] = Field(
        min_length=1,
        description=(
            "A list of all the categories to which the clothing item whose "
            "description is provided, belongs."
        ),
    )


class MatchedImageId(BaseModel):
    image_id: int = Field(..., description="Id of the best matched image")
