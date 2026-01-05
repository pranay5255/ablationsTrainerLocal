import asyncio
import argparse
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

from preprocessing.data_loader import DataLoader, DataSource
from .llm_client import LLMClient, LLMConfig
from .trace_generator import TraceGenerator
from .delta_pair_creator import DeltaPairCreator
from .rl_prompt_filter import RLPromptFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataGeneration")

async def main():
    parser = argparse.ArgumentParser(description="Smart Contract Data Generation Pipeline (SFT, DPO, RL)")
    
    # LLM Configs
    parser.add_argument("--openrouter-key", type=str, help="OpenRouter API Key")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1", help="vLLM Base URL")
    
    # Generation Tasks
    parser.add_argument("--task", choices=["sft", "dpo", "rl"], required=True, help="Task to run")
    parser.add_argument("--variant", choices=["THINK", "INSTRUCT"], default="THINK", help="Variant for SFT")
    
    # Data Configs
    parser.add_argument("--num-samples", type=int, default=10, help="Number of contracts to process")
    parser.add_argument("--output-dir", type=str, default="output/generated", help="Output directory")
    
    args = parser.parse_args()
    
    # 1. Setup Clients
    # We use Gemini Flash for most tasks as it's free/cheap on OpenRouter
    strong_config = LLMConfig(
        api_key=args.openrouter_key,
        model="google/gemini-flash-1.5-free",
        rate_limit_per_min=15
    )
    
    # If vLLM is available, we might use it for the weak model or local strong model
    weak_config = LLMConfig(
        base_url=args.vllm_url,
        model="qwen/qwen2.5-coder-0.5b-instruct", # Example local weak model
        is_local=True
    )
    
    strong_client = LLMClient(strong_config)
    weak_client = LLMClient(weak_config)
    
    # 2. Load Data
    loader = DataLoader(data_dir="output")
    logger.info("Loading contracts from SmartBugs...")
    contracts = list(loader.load_smartbugs())[:args.num_samples]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 3. Execute Task
    if args.task == "sft":
        output_file = os.path.join(args.output_dir, f"sft_{args.variant.lower()}.json")
        generator = TraceGenerator(strong_client)
        logger.info(f"Starting SFT generation for variant: {args.variant}")
        await generator.process_dataset(contracts, output_file, traces_per_contract=1)
        
    elif args.task == "dpo":
        output_file = os.path.join(args.output_dir, "dpo_delta.json")
        creator = DeltaPairCreator(strong_client, weak_client)
        logger.info("Starting DPO Delta Learning generation")
        await creator.process_dataset(contracts, output_file)
        
    elif args.task == "rl":
        output_file = os.path.join(args.output_dir, "rl_prompts_filtered.json")
        filterer = RLPromptFilter(strong_client)
        logger.info("Starting RL prompt filtering")
        await filterer.process_dataset(contracts, output_file)

    logger.info("Data generation complete.")

if __name__ == "__main__":
    asyncio.run(main())

