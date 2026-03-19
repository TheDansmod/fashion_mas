# FRAG: The Fashion Recommendation Agent
A multi-modal, multi-agent RAG framework for getting clothing item recommendations from a product catalogue.

# Demo
https://github.com/user-attachments/assets/cc40f339-4bd2-4075-aea9-9388405977f5

# Features
1. A stateful multi-agent pipeline using LangGraph with conditional branching and specialised nodes for intent parsing, visual feature extraction, retrieval, self-correction, and explanation.
2. A **self-correcting RAG** loop inspired by Self-RAG, where a critique agent evaluates retrieval quality and feeds structured correction signals back into the generation stage."
3. Uses a domain-specific multi-modal embedding model (marqo-fashionSigLIP) for cross-modal semantic search, enabling image-to-image and text-to-image retrieval across a 260k-item vector database - which acts as an analogue to the product catalogue.
4. A production-grade **MCP server** (with streamable HTTP transport) exposing a Qdrant vector database as LLM-consumable tools, following the Model Context Protocol standard."
5. Incorporates stateful multi-turn conversation with cross-turn image reference tracking and SQLite-backed checkpoint persistence for session resumability.
6. Async-first agentic backend with real-time streaming updates to a **Chainlit UI**, including per-session token usage monitoring for API rate limit management, along with **LangSmith observability** for enhanced debugging.

# Agent Orchestration Diagram
<p align="center">
  <img width="155" height="422" alt="fashion_agent_diagram" src="https://github.com/user-attachments/assets/79ef3f6a-9411-4232-b967-7272b5946b9c" />
</p>

# Install and Run Instructions
1. Clone git repo
2. Run `uv sync` from root directory
3. Create the folder for the vector db: `mkdir -p data/qdrant_storate/`
4. Install podman / docker and run `podman run -d --name qdrant-server -p 6333:6333 -p 6334:6334 -v "$(pwd)/data/qdrant_storage:/qdrant/storage:z" docker.io/qdrant/qdrant:latest` to download and setup the docker container. This is only needed when creating the container for the first time.
5. If the container already exists, just need to run `podman start qdrant-server` in order to start it.
6. Download the dataset using the kaggle CLI: `kaggle datasets download -d bothin/fashiongen-validation` or download from [here](https://www.kaggle.com/datasets/bothin/fashiongen-validation) and save to `DATAPATH`.
7. Setup the Qdrant Vector DB: `uv run cli_main.py data.vector_db.recreate=true data.fashion_gen.hdf5_path=DATAPATH`. (Say YES when it asks if you want to create / re-create the Vector DB. This should take around 30 minutes with a GPU.
6. The project uses the Mistral API by default, so need to have a `.env` file at root level with `MISTRAL_API_KEY=<key>`
7. Start the mcp server: `uv run src/mcp_server/mcp_server.py --datapath DATAPATH`
8. Start the chainlit UI: `uv run chainlit run app.py` or run the application in the terminal - without chainlit: `uv run cli_main.py`. If using the UI, it should start up in your default browser.

