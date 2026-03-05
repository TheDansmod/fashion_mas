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

# Multi-agent Architecture
This is just a first pass version of the agentic system.
1. Vision Agent: Receives image(s) and modifier text. Generates detailed description of the image.
2. Modifier Agent: Uses the description of the image and the modifier text to generate the description of the clothing that the user wants.
3. Recommendation Agent: Uses the description of the clothing to fetch images of clothing from the vector database that are a close match.
4. Explanation Agent: Explains how the recommended clothing items successfully fulfil the request.
5. State needs to contain:
    1. Modifier text received as input
    2. Image description
    3. Required clothes description
    4. Recommended clothing image 
    5. Recommended clothing description
    6. Input image(s)

# Timestamped Updates
1. 2026-03-01 12:43 First commit to github. Just added some exploration files to see how to work with qwen. Have downloaded the fashion-gen dataset, but it is using an h5 file format. Will next be trying to figure out how to work with it.
2. 2026-03-01 13:27 Explored a bit on how to work with the dataset. Checked what values are available for metadata. Will now be trying to build the vector datastore with llama index and qdrant.
3. 2026-03-02 10:36 It seems llama-index does not really provide the benefit or low-friction setup I was assuming it does. Multi-modal vector embedding does not seem like a first class citizen in its design. For now, I will be going ahead with just lang-graph and qdrant without llama-index. I will now be trying to figure out what sorts of queries I will need to make and how to setup qdrant vector database.
4. 2026-03-02 11:17 I am not sure how to integrate the embedding model into qdrant or lang-graph. I tried with the HuggingfaceEmbeddings but that did not work (apparently they need to be sentence-transformers). I also tried to check if open clip provided marqo-fashionSigLIP as a pretrained option, but they apparently do not. I am now trying to figure out how to perhaps create a class with the right interface to integrate correctly with these libraries.
    1. The langchain link says the Embeddings interface is for text models. Will have to see how to work with image models.
5. 2026-03-03 16:59 Have written code to populate the vector db. Was able to figure out how to insert stuff into the qdrant db - for both images and text. For now, have discarded the use llama index since it just seems like it tries to do both of what qdrant and langgraph do, but worse than either. Will now be running the code to populate the vector db, which might be a little slow - but is only a one-time task - hopefully. Except I haven't added code for msrp inclusion which I wanted. I will have to re-create the DB if I really want it, so I'll add it now.
6. 2026-03-03 17:24 Added msrp. Doing generation of qdrant collection now. Having to work with the hyperparameters a little since it is a really large dataset.
7. 2026-03-03 17:31 It seems insertion (upsert specifically) is really slow. Based on current estimates, it is going to take around 12 hours to insert everything into the collection. I was inserting the description text and the image itself into the payload since I did not want to fetch them from the hdf5 file. But given how long it seems to be taking, it might be prudent to do the fetch (for image and description if needed), in real time, from the hdf5 database since it allows reading arbitrary indices without loading the whole thing into memory.
8. 2026-03-03 18:26 Just by removing the image from the payload - the description and everything else remains, and by decreasing the `data_fetch_batch_size` from 1024 to 512 (might not even be needed), I was able to decrease the time neeeded to create the whole collection to 52 minutes (from 11 / 12 hours). I will try larger `data_fetch_batch_size`.
9. 2026-03-03 20:30 Completed inserting points into the collection. Not inserting the images made all the difference.
10. 2026-03-05 09:34 I have written most of hte code for a first pass agentic system. I am testing it and working out the bugs. I have also created a HuggingFace Token since they were saying that 
11. 2026-03-05 10:53 I accidentally deleted the existing collection. I will have to recreate it and it will take at least an hour to do it. Should have created a copy so that this would have been avoided - it was only 20 something MB (It is actually 80 MB).
12. 2026-03-05 14:55 I have added persistence - but I haven't been able to test whether it works. I am running into issues of CPU offloading from llama and am trying to figure out how to fix it.
    1. One approach is to decrease the context length (`num_ctx`). Current context length can be determined by initialising ollama (`ollama serve`), running the relevant model (`ollama run qwen3-vl:8b-thinking` - and we can get the list of models by running `ollama ls`), and checking out the processes (`ollama ps`) - which has a CONTEXT column which is the context length of the model. For me, it is currently 4096. It can be decreased to 2048.
    2. Another approach is to enable KV-Cache quantization. This value is `f16` by default, and can be set to `q8_0` or `q4_0` using the `OLLAMA_KV_CACHE_TYPE` environment variable, and enabling flash attention. For my model, when running it, one of the logs says KvCacheType:`blank`, which I interpret to mean it is default - which is `f16`. In order to use quantised kv cache, need to provide the environment variable before serving ollama: `OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve`. When I see the log from `ollama serve` I see that flash attention is enabled.
    3. Another approach is to force all layers to the GPU but this might lead to OOM issues. Again have to do this before serving: `OLLAMA_NUM_GPU_LAYERS=9999 ollama serve`. Or through the model file `num_gpu` parameter (but this is not mentioned in the docs).
    4. Another approach is to reduce the quantization of the model weights. If I am running `Q8_0` or `Q6_K`, I can switch to `Q4_K_M`. But it is very likely I am already using `Q4_K_M` since that is what is mentioned for the default (without thinking) model on ollama.
    5. When I checked again with Gemini, it now says that the issue is due to a bug in ollama where the ViT model remains on CPU despite there being space. But I was not sure if it was saying the truth so I have served ollama with `q8_0` KV Cache quantization and will check what happens.
13. 2026-03-05 18:08 I am frustrated with the fan not starting on linux on my laptop. I need to wait till April to (maybe) get fan control support on linux for my laptop. I am going to try and put the fan on full force from windows, restart the PC, switch to linux, and run the code then. Or maybe I will try running the code on windows itself.
14. 2026-03-05 19:17 The restarting trick did not work. The fan switched off in a few minutes after startup.

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
12. `langgraph-checkpoint-sqlite` - this is used in creating sqlite checkpoints for re-run persistence of the agent.

# TODOs
1. Figure out all the categories, sub-cat, brands, etc etc - what all elements they have
2. Set batch size of marqo-fashionSigLIP model (256, 512, 1024 etc etc) in `config/data/data_01.yaml`.
3. Handle the case when there are no input images.
4. Handle the case when the uploaded images are not that of clothes.
5. The prompts might not really count as system prompts.
6. Generate more than one match for each requirement, then ask the llm to check which of them matches
7. Handle the case when the user asks for k clothing items matching one description
8. Apply filters to match requests (also see how to incorporate filters throughout the pipeline).
9. Remove start and stop indices from the config and the vector db writer code

# Google Colab Instructions:
1. Ensure t4 runtime
2. `git clone https://github.com/TheDansmod/fashion_mas.git`
3. Move into `fashion_mas` folder and run `uv sync`
4. Need to upload qdrant collection (zip first) `zip qdrant.zip -r qdrant_multimodal_db`
5. Neeed to upload my shirt
6. Unzip (`unzip qdrant.zip`)
6. Need to ensure recreate is false (`config/data/data_01.yaml`) and ensure resume from checkpoint is false (`config/rag_pipeline/rag_pipeline_01.yaml`)
7. Install ollama `curl -fsSL https://ollama.com/install.sh | sh` after installing zstd (`sudo apt-get install zstd`)
8. Pull model: `ollama pull qwen3-vl:4b-thinking`
9. Upload .env file or create it there
10. Upload kaggle.json (legacy api key) to .kaggle/ folder
11. Run `kaggle datasets download -d bothin/fashiongen-validation`
