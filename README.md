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
7. Evaluation pipeline which uses the **LLM-as-a-judge framework** to evaluate the agent across multiple different query types including "items that go well with X", "same item, different style", "items that look good on me", and "aesthetic from image"; then judges the output of the agent on various criteria like visual grounding, individual item suitability, completeness, and coverage; then assigns a weighted final score and produces the mean and standard deviation of all final scores across the entire evaluation dataset.
8. Current Evaluation Score: **9.37 ± 0.78 / 10.00**

# Agent Orchestration Diagram
<p align="center">
  <img width="155" height="422" alt="fashion_agent_diagram" src="https://github.com/user-attachments/assets/79ef3f6a-9411-4232-b967-7272b5946b9c" />
</p>

# Install and Run Instructions
1. Clone git repo
2. Run `uv sync` from root directory
3. Chainlit does not play well with cloning / symlinking / hardlinking which is the default behaviour of uv. So need to remove chainlit and add it in copy mode:
    1. `uv remove chainlit`
    2. `uv add chainlit --link-mode copy`
4. Create the folder for the vector db: `mkdir -p data/qdrant_storate/`
5. Install podman / docker and podman-compose / docker-compose, and run `podman-compose up -d` to download and setup the docker containers.
6. Run `podman-compose down` to terminate the containers when done.
7. Download the dataset using the kaggle CLI: `kaggle datasets download -d bothin/fashiongen-validation` or download from [here](https://www.kaggle.com/datasets/bothin/fashiongen-validation) and save to `DATAPATH`.
8. Setup the Qdrant Vector DB: `uv run cli_main.py data.vector_db.recreate=true data.fashion_gen.hdf5_path=DATAPATH`. (Say YES when it asks if you want to create / re-create the Vector DB. This should take around 30 minutes with a GPU.
9. The project uses the Mistral API by default, so need to have a `.env` file at root level with `MISTRAL_API_KEY=<key>`
10. Start the mcp server: `uv run src/mcp_server/mcp_server.py --datapath DATAPATH`
    The MCP server runs on port 9000
11. Start the chainlit UI: `uv run chainlit run app.py` or run the application in the terminal - without chainlit: `uv run cli_main.py`. If using the UI, it should start up in your default browser (`http://localhost:8000`).
12. If instead, you want to run the evaluation pipeline - which evaluates the agent across multiple different query types using 60 queries and then, through the llm-as-a-judge framework, judges those queries across multiple critiera, and finally provides the mean score out of 10 and the standard deviation: `uv run cli_main.py eval.eval_mode=true`


# AWS Deployment Considerations
1. KNOWN: Embedding model - lives on the EC2 t3.large - can use Amazon Nova Embedding - but quality might be lower
2. KNOWN: MCP server - AWS Lambda + API Gateway - each tool is a lambda function - use API Gateway to be called by agent - The core flow would be: API Gateway HTTP API → Lambda (FastMCP + Mangum) → External services. Might not actually need API Gateway since you are invoking the lambda internally from EC2
3. Agent Orchestration - this runs on the EC2 t3.large
4. Chainlit UI - this runs on the EC2 t3.large - points 3. 4. are being suggested to run together EC2 t3.small - which will cost around 15 / month - for always on. But perhaps you can have it on demand in some way - if you are willing to wait for the launch time. - also need to see how to setup static IP - that might cost extra
6. KNOWN: Qdrant Docker Container - Can run on ECS Fargate 0.25vCPU and 1 GB RAM (or EC2 t4g.micro) - it will have 90 second startup time and 1-2 second query fetch time
7. KNOWN: Qdrant Vector DB - this can live on an EBS, although EFS is better (more support with Fargate) but more expensive
8. KNOWN: Fashion-Gen 14 GB database - use a single file (hdf5) entry on AWS S3 Standard - this supports random access with an object - see exploration folder
9. KNOWN: Sqlite checkpointer db - use dynamo DB
10. KNOWN: LLM Calls - AWS Bedrock - this is a very in-demand service - might have to use Mistral Medium directly for now since Bedrock does not have this model and others cost money - there is a simple ChatBedrockConverse API on LangChain which allows easy switching from Mistral to any Bedrock model
11. .env - AWS Secrets Manager / Parameter Store

Can use Qdrant Cloud Free Tier to host the docker container and the Qdrant DB - they have 0.5vCPU, 1 GB RAM, 4 GB disk space free forever

## Changes to MCP Server
1. Need to switch out code components that assume local stuff
```python
# Replace FashionSigLIPEmbedding with:
class RemoteEmbedder:
    def get_text_embedding_batch(self, texts):
        response = httpx.post("https://your-embedding-api/embed", json={"texts": texts})
        return response.json()["embeddings"]

# Replace get_fashion_gen_data with:
def get_fashion_gen_data(from_idx, to_idx):
    response = httpx.get(f"https://your-data-api/fashiongen?from={from_idx}&to={to_idx}")
    return response.json()

# Change Qdrant URL from localhost to your external instance:
QdrantConnector(url="https://your-qdrant-cloud-url", collection_name="fashion_gen")

# Replace mcp.run() at the bottom with:
from mangum import Mangum
app = mcp.http_app()   # FastMCP exposes the underlying Starlette app
handler = Mangum(app)  # this is your Lambda handler
```
2. Use asyncio.gather for all the matches together rather than each match sequentially


TODO: you have added psycopg[binary] as a dependency - but the binary is not recommended for production - please libpq on the system in a production environment.



## Injection sites:
1. all of common utils - DONE
2. rag pipeline create checkpointer provider - DONE
3. inject the llm models and the checkpointer provider itself  - DONE
4. and perhaps inject the callback config too? - NOT DOING
5. perhaps the vector db client should also be a singleton



## Files to update:
1. `app.py`                                                     DONE
2. `cli_main.py`                                                DONE
3. `src/data_manager/dynamo_db_connector.py`                    NOT-FIXING
4. `src/data_manager/session_data_table.py`                     NOT-FIXING
5. `src/data_manager/vector_db_reader.py`                       DONE
6. `src/data_manager/vector_db_writer.py`
7. `src/evaluation/llm_as_judge.py`                             DONE
8. `src/exploration/aws_upload_download_s3.py`                  NOT-FIXING
9. `src/exploration/chainlit_auth_persistence/app.py`           DONE
10. `src/exploration/hdf5_size_estimation.py`                   NOT-FIXING
11. `src/exploration/uvicorn/client.py`                         NOT-FIXING
12. `src/exploration/uvicorn/my_client.py`                      NOT-FIXING
13. `src/exploration/uvicorn/my_server.py`                      NOT-FIXING
14. `src/exploration/uvicorn/server.py`                         NOT-FIXING
15. `src/mcp_server/mcp_client.py`                              NOT-FIXING
16. `src/mcp_server/mcp_server.py`                              DONE
17. `src/rag_pipeline/checkpointer.py`                          DONE
18. `src/rag_pipeline/llm_schemas.py`                           DONE
19. `src/rag_pipeline/rag_agent.py`                             DONE
20. `src/utils/common_utils.py`                                 DONE
21. `src/utils/mock_llm_agent.py`                               DONE
22. `src/utils/model_factory.py`                                DONE
23. `src/utils/ui_node_updates.py`                              DONE


## Config folders to map
1. data             DONE
2. eval             DONE
3. exploration      DONE
4. models           DONE
5. prompts          DONE
6. rag pipeline     DONE
7. tracking         DONE
