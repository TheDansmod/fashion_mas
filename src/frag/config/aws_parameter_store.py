"""Config for using the AWS Parameter Store to manage all the configurations."""
from pydantic import BaseModel, ConfigDict

class AWSParamStoreConfig(BaseModel):
    """Config for AWS Systems Manager Parameter Store."""
    model_config = ConfigDict(frozen=True, validate_default=True)

    # all the param store parameters are prefixed with this
    default_ssm_prefix: str = '/frag/'

    # the file where to store the config obtained from the config files
    default_plan_file: str = 'src/frag/exploration/ssm_plan.yaml'

    default_region: str = 'us-east-1'

    # must match the settings config dict - env nested delimiter
    env_nested_delimiter: str = '__'

    # these are the ones to skip
    skip_configs: list[str] = ['models__llm_provider', 'logs__log_level', 'logs__is_dev', 'orchestration__checkpointer__dynamodb__profile_name', 'orchestration__checkpointer__postgres__dsn']

    # these are the ones which should be secret strings
    secure_string_configs: list[str] = ['env__hf_token', 'env__langsmith_api_key', 'env__mistral_api_key', 'env__postgres_user', 'env__postgres_password']
