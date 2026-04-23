"""Local tools, setup as MCP server alternative."""
import json
from langchain.tools import tool
from loguru import logger as log


class ProductCatalogueTools:
    def __init__(self, connector, embedder, product_categories):
        self._connector = connector
        self._embedder = embedder
        self._product_categories = product_categories

        # tools
        self._semantic_search = tool(self.semantic_search)
        self._get_datapoint_by_index = tool(self.get_datapoint_by_index)
        self._get_product_categories = tool(self.get_product_categories)

    def _reformat_image_data(self, matches):
        matched_images = []
        for match in matches:
            metadata = {
                "price": match["price"],
                "category": match["category"],
                "description": match["description"],
                "id": match["id"],
                "score": match.get("score", 0),
            }
            matched_images.append({"type": "text", "text": json.dumps(metadata)})
            matched_images.append({"type": "image", "base64": match["image"], "mime_type": "image/jpeg"})
        return matched_images

    async def semantic_search(
        self, description: str, categories: list[str], num_matches: int
    ):
        """Get num_matches images and their metadata that match the description and categories.

        Given some text description of an item of clothing or an accessory, this
        function returns `num_matches` images along with their metadata (which
        includes id, price, category, description, and score) from within the images
        that belong to the listed categories.

        Args:
            description (str): The description to which the returned images are matched.
            categories (list[str]): The matched images will belong to one of these
                listed valid categories.
            num_matches (int): The number of matched images that should be returned.

        Returns:
            Matched Images and their metadata. The metadata is a dictionary with the
            keys of `price`, `category`, `description`, `id`, and `score`. The score
            tells how good of a match (to the input text) the returned item is.
        """
        log.debug("semantic_search tool call made")
        for category in categories:
            if category not in self._product_categories:
                return {"error": f"{category} is not a valid category."}
        embedding = await asyncio.to_thread(self._embedder.get_text_embedding_batch, [description])
        embedding = embedding[0]
        matches = await self._connector.get_image_matches(
            embedding, categories=categories, num_matches=num_matches
        )
        log.info("returning some matches.")
        return self._reformat_image_data(matches)

    async def get_datapoint_by_index(self, index: int):
        """Get a datapoint, including image and metadata, using index in db."""
        from frag.data_manager.dataset_read_write import get_fashion_gen_data

        log.debug("get_datapoint_by_index tool call made")
        data = await get_fashion_gen_data(index)
        return self._reformat_image_data([data])

    def get_product_categories(self) -> list[str]:
        """Returns a list of valid product categories."""
        log.info("get_product_categories tool call made")
        return self._product_categories

    def get_tools(self):
        return [self._semantic_search, self._get_datapoint_by_index, self._get_product_categories]
