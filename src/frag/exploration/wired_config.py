from dependency_injector.wiring import inject, Provide as PV
from frag.config.container import Container

cfg = Container.config.provided

@inject
def check_wiring(
    hf_token: str = PV[cfg.env.hf_token],
    bucket_name: str = PV[cfg.exploration.s3_bucket_name],
    provider = PV[cfg.models.llm_provider],
    llm_model = PV[Container.llm_model.provided],
):
    print(hf_token)
    print(bucket_name)
    print(provider)
    print(llm_model)


