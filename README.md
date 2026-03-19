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
15. 2026-03-09 09:20 I have been able to successfully run the code on Windows and have added langsmith observability. I will now be modifying the code to incorporate some of the identified drawbacks (no image) and have some self-corrective setup.
16. 2026-03-09 13:28 I have got the new agentic workflow to run end-to-end, but there were several issues:
	1. The modifier node output the following requests:
		1.  (1512) A pair of slim-fit, dark indigo denim jeans with a high-rise waistband and a straight leg silhouette that accentuates the shirt's tailored fit, ideal for a professional or semi-formal setting.
		2. (379) A classic straight-leg, medium-wash denim jean with a comfortable stretch and a slightly tapered ankle, providing a versatile option that works well with a variety of shirt styles.
		3. (870) A bootcut denim jean in a black wash with a modern, wide leg and a subtle fade, designed to add a touch of casual elegance to the shirt's overall look.
	which successfully satisfies the requirements, but the recommender node matched clothes with the following descriptions:
		1. Relaxed-fit silk blouse in black. Scoopneck collar. Dot print at front in white. Tonal lace panels at front. Vented at sides. Lined front panel. Tonal stitching.
		2. Cotton denim biker jacket in robin-egg blue, beige, off-white and khaki. Band collar with concealed snap closure. Convertible double zip closure at front with snap-down lapel. Concealed zip pockets at front. Snap closures at waist. Two pull-loop cinch straps at sides. Ribbed upper sleeve. Contoured hem at back. Fully lined. Contrast stitching in khaki.
		3. Short sleeve t-shirt in black. Crewneck collar. Dolman sleeves. Contrast panel at interior arms in white. Tonal stitching.
	These matches are completely different from the requirements.
	2. The explanation model tried to make its own suggestions. I have now instructed it to give only explanations and if not then a rating of how well it matches - rating is not needed but will help me see how well the verifyer might work.
	3. The intent node was giving too many suggestions (not just what to extract from the image, but also what jeans to select) and it was too long. I have asked it to be concise (5 sentences or less) and have asked it to restrict itself to image extraction requirements.
17. 2026-03-09 18:40 I have tried to solve all of these above issues - 2 and 3 through prompt engineering - and 1 through added a filtration node based on the category of the item. Will be testing it out in windows.
18. 2026-03-09 19:24 I ran the code on windows, but faced issue with getting the filtration output to match the valid categories - the execution gets hung up on that LLM call for quite a while. I am not sure if the single LLM call was taking a while or if langgraph was doing retries and that was taking time. I tried adding a timeout (`client_kwargs={"timeout": 30.0}`), but it was not respected. I will now be trying to use indexed arrays.
19. 2026-03-09 22:04 It seems like the qdrant client is returning random fucking shit despite me telling it to return only things that match the specific category. I asked it for jeans 3 times and it returned non-jeans items id 340 (pants), 791 (sweaters), 870 (tops). This is independent of the issue of trying to get the LLM model to output some subset of 49 categories.
20. 2026-03-09 22:17 I am tired for today. Will continue investigating this shit tomorrow. Regarding the issue with the recommender not doing filtering I think it might just be an issue of not passing the categories.
21. 2026-03-10 10:45 I was able to fix the issue with the recommender not suggesting aligned category results. It is better now. But I still need to work on getting the filtration node to actually work. I think I am going to try hierarchical categories.
22. 2026-03-10 11:11 Part of the issue with hierarchical categories is that it is difficult to create mutually exclusive super-categories, and additionally, the existing categorization is imperfect - having unnecessary overlaps and repetition. I think I am just going to try and ask it to output a simple string and then I'll parse that string to see if it contains any of the categories.
23. 2026-03-10 12:46 The results from the LLMs seem to be pretty good and nearly exactly what I want, but it seems like even though the category of the item is jeans, the actual item is a jacket. I might have to iterate through the entire dataset and assign new categories to each item if that is actually the case. Am checking if that is the case.
24. 2026-03-10 13:43 There is a discrepancy between the index of the point in the qdrant database and the index of the same datapoint in the hdf5 database, which is causing the issue. Currently, the only solution I can think of is to re-create the index, which will take a really large amount of time - 10 to 12 hours total.
25. 2026-03-10 17:21 I have re-created the qdrant vector db on linux (the issue was that I was over-writing every batch to the same indices), and that was also useless since it became really large (4 GB) and I got a warning saying to use docker or cloud with qdrant. I have decided to go to docker. I have a migrate script that will hopefully work - since when I had tried to load the qdrant client before, it crashed the full program. If the migration does not work, I will be forced to re-create the qdrant db, this time on docker.
26. 2026-03-10 18:04 I tried to do the migration and it did not work, same issue of full RAM usage followed by crash. I am now trying to re-create the vector db by inserting directly into the docker container based qdrant client. - 18:35 - recreated - 25 mins for 260490 points.
27. 2026-03-10 18:44 The database seems to be returning index values which are much larger than 260490. Will have to figure out what the issue is. I made a mistake again. The sequential index of the datapoint in the h5py database has no relationship with the value of the index key for that datapoint. I will most likely have to re-create the database unless you create a static mapping between the sequential index and the key-value index. I have just iterated through the database and determined that `index_2` is actually perfectly aligned with the true sequential index. I just need to substitute one for the other.
28. 2026-03-10 20:36 I have run the code for updating the payload values without having to re-create the db by leveraging the batch update points and delete and set payload operations - am really thankful for these. It was relatively fast - no more than a few minutes. Will test now.
29. 2026-03-10 22:21 It seems like there will be errors if I try and use the existing linux docker setup in windows. As of now I am resigned to re-creating the database on windows. However, I am facing some issues on windows with regards to grpc communication. might have to switch to rest api to see if it works. It did not work even with using REST API. Need to figure this out.
30. 2026-03-11 10:20 I ran the code on windows and it seems to work after the podman fix, but it is taking an absurd amount of time. I suspect it might be the embedding generation that is slow. I will check. It is using only the CPU. Apparently on windows, just doing `uv add torch` does not include cuda support. I have made the appropriate modification to the pyproject.toml file and have installed the cuda-based pytorch version.
31. 2026-03-11 12:17 Completed generation, took around 30 minutes.
32. 2026-03-13 20:33 I am trying to switch to using Mistral AI since it has a pretty generous free tier. I am however, facing issues in that langchain's Mistral AI integration does not include any image capabilities which is making things pretty difficult since images are a central part of my product. I am also trying to sequence multiple things that might impact one another:
    1. Trying to add a server like FastAPI / Flask - this integrates with Chainlit UI and User History
    2. Trying to add a MCP server - this exposes tools and stuff that integrates with how the agent is orchestrated
    3. Agent Orchestration to allow for self-correction loops or reason and plan loops - this integrates with specifying tools for the agent to invoke and use.
    4. Switching to Mistral AI - this integrates with image based inputs, using a `create_agent` loop vs a langgraph based loop
33. 2026-03-13 21:44 The langchain chat mistral ai integration does seem to support images, but due to the rate limit, I think it only supports low resolution images which makes things a little difficult for me since I rely heavily on the generated descriptions of images. 
34. 2026-03-13 22:39 I just tested mistral with the old ollama code and things seem to have worked great. I'm thinking I won't use the create agent automated Re-Act setup since that does not demonstrate competence, but then, the goal is to quickly get something out rather than show a lot of skills.
35. 2026-03-16 11:27 I have been able to setup and test an MCP server for a shipping agent. And the Mistral AI seems to have done well on it. I am currently working on creating a MCP server and client for the qdrant client. I will later be fixing the agent orchestration.
36. 2026-03-16 17:24 I was fretting with Mistral requiring only text input as tool call results, but that is not the case - as I had expected. Even if we pass it a list of dicts with keys text and type (and should work with images identically), then it still works.
37. 2026-03-16 20:51 I was able to get the code running with the MCP server. Now I just need to make a few changes - the self-correction loop, human interaction, and the UI.
38. 2026-03-18 15:29 I am desperately trying to figure out how to integrate chainlit with the langgraph setup that I currently have, but there seems to be no end to the trouble. It either does not run or it runs multiple times messing up the langgraph loop. It also does not play well with hydra since it wants to have control over the launch. For the moment, I am going to go back to simple while loop and check if everything is working fine.
39. 2026-03-19 08:07 Things seem to be working fine with chainlit now, but the recommendations are still not very good, perhaps you need to add a critique node after all? It took around 10k total tokens for a single run with mistral.

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
13. `langchain-google-genai` - this is required to invoke Google AI APIs (I am trying with Gemma model)
14. `langchain-mistralai` - this is used to invoke Mistral's AI APIs and LLM models (they have a pretty generous free tier)
15. `langchain` - this is needed for creating my own tools
16. `langchain-mcp-adapters` - this is used to create a mcp client which can be directly used by langchain
17. `aiosqlite` - this is needed for asynchronous sqlite usage
18. `chainlit` - this is for the UI

# TODOs
4. Handle the case when the uploaded images are not that of clothes.
5. The prompts might not really count as system prompts.
6. Generate more than one match for each requirement, then ask the llm to check which of them matches
7. Handle the case when the user asks for k clothing items matching one description
8. Apply filters to match requests (also see how to incorporate filters throughout the pipeline).
9. Remove start and stop indices from the config and the vector db writer code
10. Finalised improvement plan:
    1. LLM Observability - langsmith
    2. Self-Correcting Agent (CRag / SelfRAG) papers
    3. Qdrant as MCP server

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

# Windows Changes:
1. Create folder `fashion_mas_windows` and clone the repository into it
2. Install uv: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
	1. To uninstall, run the below commands
		uv cache clean
		rm -r "$(uv python dir)"
		rm -r "$(uv tool dir)"
	2. Then run the below commands
		rm $HOME\.local\bin\uv.exe
		rm $HOME\.local\bin\uvx.exe
		rm $HOME\.local\bin\uvw.exe
3. To install Ollama: `irm https://ollama.com/install.ps1 | iex` (to uninstall it go to installed apps and choose ollama from there) - ollama is automatically setup as a run-on-starup app.
4. Run the `uv sync` command from the `fashion_mas` folder.
5. Changed the path of the hdf5 dataset to be a windows path rather than a linux path - ensure you use single quotes rather than double quotes - single quotes means characters are literals, else they might be interpreted as escape sequences.
6. Created the data folder and put the my shirt in that folder from the original `fashion_mas` folder.
7. Install the model: `ollama pull qwen3-vl:4b-thinking`
8. Apparently there is an issue with trying to migrate qdrant folder directly from linux to windows, so we will have to re-create the collection.
9. Copy the .env file from `fashion_mas` folder (original)
10. Ensure recreate (`data_01.yaml`) is true, resume from checkpoint (`rag_pipeline_01.yaml`) is false, set main.py to run the populate vector db command - have re-created the db.
11. Ensure re-create (`data_01.yaml`) is false, resume from checkpoint (`rag_pipeline_01.yaml`) is false, set main.py to run fashion agent.
12. I added some .env variables to the file in original `fashion_mas` (I am getting frustrated jumping between linux and windows - for now I am going to stay on windows), so I copied over the file to this project.
13. The Langsmith thing was not working. I have created new key.
14. Still not working. Now giving errors about internet connection 10053 Connection Aborted Error. I have added a new key for the endpoint in the EU. Let' try this way. Works.
15. I only have 5k traces per month, so I will not log most development traces.
16. Set `$env:PYTHONUTF8=1` to ensure that the console can handle emoji outputs.
17. For the docker setup: install the podman exe file, run it, run `podman machine init` on your system, then run `podman machine start` (`podman machine stop` to terminate the podman guest)
18. Create the qdrant container image (no :z) with the volume mount of its own - using named volume for windows (from qdrant docs https://qdrant.tech/documentation/quickstart/):
	1. `podman volume create qdrant_storage_volume`
	2. `podman run -d --name qdrant-server -p 6333:6333 -p 6334:6334 -v "qdrant_storage_volume:/qdrant/storage" docker.io/qdrant/qdrant:latest`

# Windows Changes (except setup parts - after git pull - after initial qdrant creation):
This is after I have already run the code on windows, but have done some development on linux after that, and am switching back to windows to run the code.
1. Change the hdf5 file path to be windows sensitive: `'C:\Users\lordh\Documents\Svalbard\Data\fashion-gen\fashiongen_256_256_train.h5'` - single quotes are mandatory

# Compare files in two folders:
```
diff -yr /mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas_windows/fashion_mas /mnt/windows/Users/lordh/Documents/LibraryOfBabel/Projects/fashion_mas -X /mnt/windows/Users/lordh/Documents/LibraryOfBabel/Miscellany/Danish/linux_temp/temp_files/0427_fashion_mas_compare_exclude.txt --strip-trailing-cr --suppress-common-lines
```

# Checklist before running code:
1. Re-create vector database (data 01.yaml)
2. Resume from checkpoint / thread id (rag pipeline 01.yaml)
3. LLM model - qwen / mock / mistral (models 01.yaml)
4. Langsmith API enabled / not (.env)
5. Ensure podman container running (podman ps)
6. If using mcp client, ensure server is running
7. If not doing testing, ensure that no mocking of `create_agent` is present in rag agent

# Qdrant with Docker / Podman
After I fixed the issue where only a small number of points were constantly being over-written while creating the qdrant collection, the size of the collection ballooned to 4 GB. I also got a warning saying that I should use Qdrant on docker or cloud with so many points / vectors.
1. Create the docker container:
`podman run -d --name qdrant-server -p 6333:6333 -p 6334:6334 -v "$(pwd)/data/qdrant_storage:/qdrant/storage:z" docker.io/qdrant/qdrant:latest`
where 6333 is the rest api, 6334 is the grpc api; :z applies a shared SELinux label; we are creating a volume in the `./data/qdrant_storage` folder.
2. Stop the docker container gracefully: `podman stop --time 30 qdrant-server`
3. Start the docker container: `podman start qdrant-server`

# Debug Windows Podman connectivity Issues
1. I want to figure out if the issue is with podman on windows or if it is with the qdrant container setup. I will try with traefik/whoami.
2. Running command: `podman run --detach --name network-diagnostic --publish 8080:80 docker.io/traefik/whoami`
3. Executing `curl.exe --verbose --ipv4 http://127.0.0.1:8080` gives an error with reset. It seems the issue is with podman on windows.
4. I stopped all running containers, `podman machine stop`, `podman machine rm`, `wsl --shutdown`.
5. Settings for WSL can be seen by going to Start Menu > WSL Settings App (I was not able to find the .wslconfig file)
6. I turned the Hyper-V Firewall Enabled to off in the WSL Settings
7. I setup podman again: `podman machine init`, `podman machine start`, then command 2 again - whoami, again tried curl (command 3) - but same connection error
8. I tried to do `wsl hostname -I` and `wsl --distribution podman-machine-default hostname -i` but neither worked since hostname command was not defined (likely due to the pared down distribution of podman)
10. I set the podman machine to rootful: `podman machine stop`, `podman machine set --rootful`, `podman machine start`, command 2, command 3 - IT WORKED
11. Will remove the whoami image and try with qdrant
12. Fix for podman network issues on windows: use rootful setup rather than rootless

# Correcting Windows / Linux CRLF vs LF
1. Create a `.gitattributes` file, add line: `* text=auto eol=lf`
2. `git add --renormalize .`, do a commit and push
3. refresh your local un-normalised files: `git rm --cached -r .`, `git reset --hard HEAD`
4. Verify: `git ls-files --eol`


# Branches and their purposes
1. `ollama_to_mistral` - the rag agent uses ollama, adapt it to use mistral - things like rate limiting, smaller image sizes etc - just to see the results

# Learnings
## MCP
1. The MCP protocol acts in a client and server setup. The server defines tools (get weather / calculator), resources (database etc), and prompts which can be used by the MCP Client. The server also executes the tools and returns their results.
2. The Client is an application like Claude Code or some other script. The client acts as an intermediary between the server and the LLM.
3. The Flow:
    1. The client sends a request to the server to figure out the available tools.
    2. The client tells the LLM "these are the available tools" - and this may be all the truly available tools or it might just be a subset, and gives the LLM some task.
    3. The LLM then says to the client "Ok, run these tools with these parameters" and the client then forwards them to the Server
    4. The Server executes the requested tools and returns the results to the Client.
    5. The Client then passes these on to the LLM and waits for what to do next.
4. The entire setup is built on async calls
5. JSON-RPC is used as the wire protocol over (possibly):
    1. Stdio - The host / client process spawns the MCP server as a child process and communicates by writing to the child's stdio and reading from the child's stdout. It is really fast. The limitation here is that one host can only connect to one server process and nobody else.
    2. SSE (depreciated) - Server Sent Events - it used HTTP which is unidirectional - but creates two permanent endpoints - for sending and receiving - this would require session management across the two connections which is difficult
    3. Streamable HTTP - This uses just one HTTP endpoint. The client can sends messages via POST requests and accepts both a JSON response (short reponses) or persistent connection where the server streams back the response over Server Sent Events (SSE). To allow the server to push information to the client, the client can open a persistent GET request.

## Langchain
1. Can't do both bind tools and structured output together - have to specify tools differently - this is since structured output acts as another bind tools and that over-writes the bind tools call


## Qdrant MCP Server Code:
1. `main.py` - just sets the transport - stdio, sse, streamable http from cli args, and invokes `server.py`
2. `server.py` - sets up tool settings, qdrant settings, embedding provider settings and invokes the core part - the Qdrant MCP Server
3. The default embedding provider used is sentence transformer mini
4. `mcp_server.py` - conatains the QdrantMCPServer - the init fn sets up the embedding provider, the connection to the qdrant server - either local or through url and an async qdrant client with store and search functionality with filters, and calls the setup tools function
5. `setup_tools` in `mcp_server.QdrantMCPServer` - defines two async functions find and store (those actually available as tools) which take in the context and the search string or the data to save - the string and metadata. it invokes the store and search functions from the qdrant connection after using the embeddings
6. I am not 100% certain about this, but I think before invoking these guys we need to setup the QdrantSettings in some way - the provided code is just a template.
