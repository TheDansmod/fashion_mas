# Goal
1. To create a multi-agent system that can do fashion recommendation
2. It should be able to take as input both images (of clothing items) and text (description, modifiers, etc) and search a product database and produce recommended items and explain why it made those recommendations

# Implementation Details
1. We will be using the Fashion-Gen dataset in an MCP server as our product database (the MCP server will come later, for now, just langgraph and Qdrant)
2. I will be using Langgraph for agent tooling, hydra for configuration management, mlflow for experiment tracking
3. I will be using the Qdrant Vector DB for multi-modal vector storage
4. Qwen3-VL 8B (potentially thinking) will act as the VLM agent, Gemma3-27b will act as the LLM backend, marqo-fashionSigLIP (from huggingface) will be used to generate multi-modal embeddings
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
	3. `input_brand` - str, want
	4. `input_category` - str, want
	5. `input_composition` - str, want
	6. `input_concat_description` - seems to be some weird amalgamation of descriptions. Best to ignore it for now.
	7. `input_department` - str, want
	8. `input_description` - str, want
	9. `input_gender` - str, want
	10. `input_image`
	11. `input_msrpUSD` - float32, want
	12. `input_name` - str, want
	13. `input_pose` - this is some internal string, that isn't really meaningful. Can ignore for now.
	14. `input_productID`
	15. `input_season` - This is also a little weird, but somewhat understandable - like SS2017. I don't know what SS is but 2017 is very likely to be the year. We'll still ignore for now.
	16. `input_subcategory` - str, want

# Timestamped Updates
1. 2026-03-01 12:43 First commit to github. Just added some exploration files to see how to work with qwen. Have downloaded the fashion-gen dataset, but it is using an h5 file format. Will next be trying to figure out how to work with it.
2. 2026-03-01 13:27 Explored a bit on how to work with the dataset. Checked what values are available for metadata. Will now be trying to build the vector datastore with llama index and qdrant.
3. 2026-03-02 10:36 It seems llama-index does not really provide the benefit or low-friction setup I was assuming it does. Multi-modal vector embedding does not seem like a first class citizen in its design. For now, I will be going ahead with just lang-graph and qdrant without llama-index. I will now be trying to figure out what sorts of queries I will need to make and how to setup qdrant vector database.
4. 2026-03-02 11:17 I am not sure how to integrate the embedding model into qdrant or lang-graph. I tried with the HuggingfaceEmbeddings but that did not work (apparently they need to be sentence-transformers). I also tried to check if open clip provided marqo-fashionSigLIP as a pretrained option, but they apparently do not. I am now trying to figure out how to perhaps create a class with the right interface to integrate correctly with these libraries.
    1. The langchain link says the Embeddings interface is for text models. Will have to see how to work with image models.
5. 2026-03-03 16:59 Have written code to populate the vector db. Was able to figure out how to insert stuff into the qdrant db - for both images and text. For now, have discarded the use llama index since it just seems like it tries to do both of what qdrant and langgraph do, but worse than either. Will now be running the code to populate the vector db, which might be a little slow - but is only a one-time task - hopefully. Except I haven't added code for msrp inclusion which I wanted. I will have to re-create the DB if I really want it, so I'll add it now.
6. 2026-03-03 17:24 Added msrp. Doing generation of qdrant collection now. Having to work with the hyperparameters a little since it is a really large dataset.
7. 2026-03-03 17:31 It seems insertion (upsert specifically) is really slow. Based on current estimates, it is going to take around 12 hours to insert everything into the collection. I was inserting the description text and the image itself into the payload since I did not want to fetch them from the hdf5 file. But given how long it seems to be taking, it might be prudent to do the fetch (for image and description if needed), in real time, from the hdf5 database since it allows reading arbitrary indices without loading the whole thing into memory.

# Library Dependency and their purpose
1. `langgraph` - agent orchestration. needed for the multi-agent system
2. `langchain_ollama` - I am using qwen local model through ollama. This is the langchain library that helps interface with it.
3. `hydra-core` - this is for ease of configuration management.
4. `hydra-colorlog` - this is for automatic colourful logging.
5. `h5py` - the fashion-gen dataset being used is in .h5 format which needs this python library to interface with it.
6. `pillow` - to convert the images from the hdf5 dataset into an image
7. `qdrant-client` - this is the vector store I will be using for both the image embeddings and the text embeddings.
8. `transformers` - need to run the marqo-fashionSigLIP model
9. `torch` - also needed to run the marqo-fashionSigLIP model
10. `open-clip-torch` - when I just ran with the above two imports, I got an error saying they wanted `open_clip` and `ftfy`. So, apparently this is also needed.
11. `python-dotenv` - this is required to load dotenv files (like for `HYDRA_FULL_ERROR=1` to get a full stack trace)

# TODOs
1. Figure out all the categories, sub-cat, brands, etc etc
2. Include msrp in some way - it is not string so isn't included in the `string_attributes` in `config/data/data_01.yaml`.
3. For batch size of marqo-fashionSigLIP model (256, 512, 1024 etc etc) in `config/data/data_01.yaml`.
