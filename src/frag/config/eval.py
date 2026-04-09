"""Config for LLM-as-a-judge evaluation."""

from pydantic import BaseModel, ConfigDict, FilePath, DirectoryPath


class JudgementWeights(BaseModel):
    """Relative weights for each judgement criteria.

    They don't necessarily have to sum to 1."""

    model_config = ConfigDict(frozen=True, validate_default=True)

    visual_grounding: float = 0.2
    item_suitability: float = 0.3
    completeness_coverage: float = 0.5


class EvaluationConfig(BaseModel):
    """Config for evaluation."""

    model_config = ConfigDict(frozen=True, validate_default=True)

    # these are the text inputs for each query
    json_queries_path: FilePath = "data/evaluation_data/eval_queries.json"

    # these are the images used in some evaluation queries - they are numbered in correspondence to the text query index they belong to
    query_images_path: str = "data/evaluation_data/eval_images/{image_filename}"

    # the folder where the images are present
    query_images_folder_path: DirectoryPath = "data/evaluation_data/eval_images"

    # this is the json file where the results of the evaluation are stored - we do evaluation and judgement of those evaluations in separate passes to keep the overall llm use low
    eval_response_file_path: FilePath = "data/evaluation_data/eval_soln.json"

    # this is the json file where the results of the judgement are stored - they contain the score breakdown and the final score
    judgement_file_path: FilePath = "data/evaluation_data/eval_judgement.json"

    # these are the relative weights for each judgement criteria - they don't necessarily have to sum to 1
    weights: JudgementWeights = JudgementWeights()

    eval_mode: bool = False
