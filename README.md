# Goal
1. To create a multi-agent system that can do fashion recommendation
2. It should be able to take as input both images (of clothing items) and text (description, modifiers, etc) and search a product database and produce recommended items and explain why it made those recommendations

# Implementation Details
1. We will be using the Fashion-Gen dataset in an MCP server as our product database (the MCP server will come later, for now, just LlamaIndex and Qdrant)
2. I will be using Langgraph for agent tooling, hydra for configuration management, mlflow for experiment tracking
3. I will be using the Qdrant Vector DB for multi-modal vector storage, and Llama Index will handle product ingestion and querying
4. Qwen3-VL 8B (potentially thinking) will act as the VLM agent, Gemma3-27b will act as the LLM backend, clip will be used to generate multi-modal embeddings
5. Streamlit or Chainlit will be used to create the product frontend
6. Regarding Agent Orchestration: There will be an Orchestrator Agent (OA) which routes to the Vision Agent (VA) and the User History Agent (UHA). The VA and UHA then send their results to the Recommendation Agent (RA) which finally sends its recommendations to the Explanation Agent (EA) which explains why the choice was made.
7. If fine-tuning is required then it can be done on the FashionRec dataset (HuggingFace PEFT with QLoRA).
8. For now, I will be building the agent without the User History part.
9. Later, it might be useful to have an agent that can generate images based on some description so that when the user asks for something different etc we can generate a few image options and use them in the rest of the pipeline.
10. Currently, the images in the dataset are only 256 x 256, later you can obtain the larger dataset which has high resolution images

# Fashion-Gen Dataset
1. It contains the following datasets:
    1. `index`
	2. `index_2`
	3. `input_brand`
	4. `input_category`
	5. `input_composition`
	6. `input_concat_description`
	7. `input_department`
	8. `input_description`
	9. `input_gender`
	10. `input_image`
	11. `input_msrpUSD`
	12. `input_name`
	13. `input_pose`
	14. `input_productID`
	15. `input_season`
	16. `input_subcategory`

# Timestamped Updates
1. 2026-03-01 12:43 First commit to github. Just added some exploration files to see how to work with qwen. Have downloaded the fashion-gen dataset, but it is using an h5 file format. Will next be trying to figure out how to work with it.
2. 2026-03-01 13:27 Explored a bit on how to work with the dataset. Checked what values are available for metadata. Will now be trying to build the vector datastore with llama index and qdrant.

# Library Dependency and their purpose
1. `langgraph` - agent orchestration. needed for the multi-agent system
2. `langchain_ollama` - I am using qwen local model through ollama. This is the langchain library that helps interface with it.
3. `hydra-core` - this is for ease of configuration management.
4. `hydra-colorlog` - this is for automatic colourful logging.
5. `h5py` - the fashion-gen dataset being used is in .h5 format which needs this python library to interface with it.
6. `pillow` - to convert the images from the hdf5 dataset into an image
