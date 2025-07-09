import asyncio
from your_arxiv_analyzer import ArxivMCPServer


async def search_related_papers():
    # Initialize the server
    server = ArxivMCPServer(debug_mode=True)

    # Analyze a paper
    paper_url = "https://arxiv.org/abs/2301.07041"  # Example: GPT-4 paper
    result = await server.analyze_paper_citations(paper_url)

    return result


# Run the search
result = asyncio.run(search_related_papers())