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

# Timestamped Updates
