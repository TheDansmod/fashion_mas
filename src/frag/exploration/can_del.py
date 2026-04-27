import asyncio
import base64
import csv
import functools
from datetime import datetime

from PIL import Image
from io import BytesIO
from loguru import logger as log
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from langchain_core.callbacks import UsageMetadataCallbackHandler

def update_token_use(usage_metadata, tracker_path=r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/data/token_usage.csv"):
    """Updates the token usage tracking csv file with the data from the callback."""
    log.info(
        f"Saving token use data for {len(usage_metadata)} models. "
        "Should be invoked just once every full run."
    )
    with open(tracker_path, "a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for model_name, metadata in usage_metadata.items():
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    model_name,
                    metadata["input_tokens"],
                    metadata["output_tokens"],
                    metadata["total_tokens"],
                ]
            )


def track_token_use(func):
    """Decorator to track token usage for LLM calls."""
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            callback = UsageMetadataCallbackHandler()
            callback_config = {"callbacks": [callback]}
            kwargs["callback_config"] = callback_config
            result = None
            try:
                result = await func(*args, **kwargs)
            except Exception as e:
                log.exception("Exception caught inside track_token_use function.")
                raise
            finally:
                update_token_use(callback.usage_metadata)
            return result

        return wrapper
    else:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            callback = UsageMetadataCallbackHandler()
            callback_config = {"callbacks": [callback]}
            kwargs["callback_config"] = callback_config
            result = None
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                log.exception("Exception caught inside track_token_use function.")
                raise
            finally:
                update_token_use(callback.usage_metadata)
            return result

        return wrapper

def encode_image(image_path=None, numpy_image=None):
    """Encode an image to base64 from file path or numpy ndarray."""
    if (image_path is None) == (numpy_image is None):
        raise ValueError("Exactly 1 of image_path or numpy_image must be provided.")
    if image_path:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    if numpy_image is not None:
        img = Image.fromarray(numpy_image)
        buffer = BytesIO()
        img.save(buffer, format="png")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_multi_image_prompt_message(image_paths, text_prompt):
    """Get langgraph compatible prompt containing an image and some text."""
    content = []
    for image_path in image_paths:
        image_data = encode_image(image_path)
        img_format = "png" if image_path.lower().endswith(".png") else "jpeg"
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{img_format};base64,{image_data}",
                },
            }
        )
    content.append(
        {
            "type": "text",
            "text": text_prompt,
        }
    )
    message = [HumanMessage(content=content)]
    return message

class LLMResponse01(BaseModel):
    num_input_images: int = Field(ge=1, le=10, description="The number of images in the input")
    image_descriptions: list[str] = Field(min_length=1, description="The list of strings, where each string is at most 3 lines long. Each string is an accurate description of the corresponding input image, in the order in which the images are found in the upload.")

@track_token_use
def test_01(model, *, callback_config=None):
    image_paths = [
        r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/data/temp_images/f97239d3-3af6-45e1-92a2-17c449c23e9e.png",
        r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/data/temp_images/f4eb9d30-cbbe-47ad-862c-79882b7696e0.png",
        r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/data/temp_images/f1fb7b85-c138-41b7-9e26-2989d0cfad10.png",
        r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/data/temp_images/f1ea4439-e7cf-4f53-a8c2-723c4b8f5b84.png",
        r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/data/temp_images/f0570326-7dee-43e1-88f5-b10e68483539.png",
        r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/data/temp_images/efed15c8-bf5b-446b-843c-cbe87b982543.png",
        r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/data/temp_images/ee3692f2-3e21-4e1a-b1cb-b261b23f18a6.png",
        r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/data/temp_images/ee2011aa-a796-4be2-b983-6b11723db87b.png",
        r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/data/temp_images/ec45d092-6c1e-4ef0-b54d-88492ee0d893.png",
        r"/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas/data/temp_images/e73312b8-8948-47d1-b5e2-4cd5926b0b69.png",
    ]
    text_prompt = "Please provide an accurate at most 3 line description of the clothing items of focus in each of the uploaded images. The output should contain the number of input images in the prompt and a list of strings where each string is an at most 3 line description of the primary clothing item in the corresponding image."
    prompt = get_multi_image_prompt_message(image_paths, text_prompt)
    structured_model = model.with_structured_output(LLMResponse01)
    log.debug("Invoking model.")
    response = structured_model.invoke(prompt, config=callback_config)
    log.info("Response from Model:\t{}", response.num_input_images)
    for idx, desc in enumerate(response.image_descriptions):
        log.info("\t{}. {}", idx + 1, desc)

def main():
    log.debug("in main method")
    model = ChatBedrockConverse(
        model_id="us.amazon.nova-2-lite-v1:0",
        region_name="us-east-1",
        temperature=0.6
    )
    test_01(model=model)

if __name__ == '__main__':
    main()
