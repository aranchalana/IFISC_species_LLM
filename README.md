# IFISC_species_LLM

A repository to create a data base based in species names. It looks for the species names, find with multiple searchers the research papers, and look for the information using local LLM.


**************** SEARCH SPECIES FROM RESEARCH PAPERS

    cd LOCAL-LLM
    Activate global-llm (env based in requirements)
    Create txt with the names of the species.
 
 
# Basic usage with Claude API
python complete_biodiversity_app_LLM.py "Xyrichtys novacula" --ai-backend claude --claude-api-key YOUR_API_KEY

# Using free local Ollama
python complete_biodiversity_app_LLM.py "Xyrichtys novacula" --ai-backend ollama

# Multiple species with custom settings
python complete_biodiversity_app_LLM.py "Xyrichtys novacula" "Coris julis" \
    --ai-backend ollama \
    --start-year 2020 \
    --end-year 2024 \
    --max-results 50 \
    --output-dir ./results

# See all options
python complete_biodiversity_app_LLM.py --help

++++++ terminal:
ollama serve
++++++ other terminal:
ollama pull phi3:mini
python biodiversity_cli_v2.py "Xyrichtys novacula" --ai-backend ollama --ollama-model phi3:mini --max-extract 10
