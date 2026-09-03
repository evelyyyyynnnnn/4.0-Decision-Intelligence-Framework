import openai
import requests
import json
import re
import random
from bs4 import BeautifulSoup


# 设置 OpenAI API Key 和 Google API Key
OPENAI_API_KEY = "your-openai-api-key"
GOOGLE_API_KEY = "your-google-api-key"
GOOGLE_CSE_ID = "your-google-cse-id"
SEARCH_ENGINE_URL = "https://www.googleapis.com/customsearch/v1"
SERPAPI_KEY = "your-serpapi-key"
MODEL_NAME = "gpt-3.5-turbo-1106"

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def ask_chatgpt(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": "你是一个智能联网搜索助手。"},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content


def search_google(query, num_results=10):
    """
    Get search results using Google Custom Search API
    """
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": query,
        "num": num_results
    }

    response = requests.get(SEARCH_ENGINE_URL, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Google API 请求失败，状态码: {response.status_code}")
        return None


def fetch_full_content(url):
    """
    Visit a web page and extract the main content
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = [p.get_text() for p in soup.find_all("p") if len(p.get_text()) > 50]
            return "\n".join(paragraphs[:5]) if paragraphs else "无法提取网页正文内容"
        else:
            return f"请求失败，状态码: {response.status_code}"
    except Exception as e:
        return f"请求错误: {e}"


def get_combined_research(query, num_results=10):
    """
    Get the first 'num_results' Google search results, extract the main content, and combine them into one block of text.
    """
    search_results = search_google(query, num_results)
    research = ""

    if search_results:
        for item in search_results.get("items", []):
            url = item.get("link", "")
            title = item.get("title", "无标题")
            
            full_content = fetch_full_content(url)  # Scrape the main content
            research += f"\n标题: {title}\n{full_content}\n\n{'='*80}\n"

    return research  # Return all scraped content


def fetch_scholar_papers(query, num_results=25):
    """
    Fetch scholarly papers related to the query from Google Scholar using SerpAPI
    """
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": num_results
    }
    response = requests.get(url, params=params)
    return response.json().get("organic_results", [])


def format_to_apa(paper, index):
    """
    Format a paper's citation in APA style
    """
    authors_list = paper.get("publication_info", {}).get("authors", [])
    formatted_authors = []
    for author in authors_list:
        name_parts = author["name"].split()
        if len(name_parts) > 1:
            formatted_authors.append(f"{name_parts[-1]}-{''.join([p[0] for p in name_parts[:-1]])}")
        else:
            formatted_authors.append(name_parts[0])
    authors = ", ".join(formatted_authors)
    
    year_match = re.search(r'\b(\d{4})\b', paper.get("publication_info", {}).get("summary", ""))
    year = year_match.group(1) if year_match else "n.d."
    
    title = paper.get("title", "Unknown Title")
    title = re.sub(r": (\w)", lambda m: f": {m.group(1).upper()}", title)
    
    summary_parts = paper.get("publication_info", {}).get("summary", "").split(" - ")
    journal = summary_parts[1] if len(summary_parts) > 1 else "Unknown Journal"
    
    doi = paper.get("link", "")
    
    return f"{authors}. ({year}). {title}. {journal}. DOI: {doi}", authors, year, index


def generate_in_text_citations(references):
    """
    Generate in-text citations based on APA format for the references
    """
    citations = []
    for ref in references:
        authors = ref[1].split(", ")
        year = ref[2]
        index = ref[3]
        if len(authors) == 1:
            citation = f"({authors[0]}, {year})[{index}]"
        elif len(authors) == 2:
            citation = f"({authors[0]} et {authors[1]}, {year})[{index}]"
        else:
            citation = f"({authors[0]} et al, {year})[{index}]"
        citations.append(citation)
    return citations


def insert_in_text_citations(text, citations):
    """
    Insert in-text citations into the research paper content
    """
    sentences = re.split(r'(\.|!|\?)', text)
    num_citations = min(len(sentences) // 3, len(citations))
    random_indices = random.sample(range(len(sentences)), num_citations)
    for i in sorted(random_indices):
        if not re.match(r'^\s*#+\s', sentences[i]) and citations:  # Ensure it's not a title
            sentences[i] = sentences[i].strip() + " " + citations.pop(0)
    return "".join(sentences)


# Main function to generate references and citations for a given research topic
def generate_references_and_citations(topic):
    # Fetch real-time research content
    research = get_combined_research(topic, num_results=10)

    # Fetch scholarly papers and format them to APA style
    papers = fetch_scholar_papers(topic)
    references = [format_to_apa(paper, i+1) for i, paper in enumerate(papers[:25])]

    # Generate in-text citations for the paper
    in_text_citations = generate_in_text_citations(references)

    # Combine research and references in the final output
    research_output = f"Research Topic: {topic}\n\nResearch Summary:\n{research}\n\nReferences (APA Format):\n"
    for ref in references:
        research_output += f"[{ref[3]}] {ref[0]}\n"

    # Return both research summary and APA references
    return research_output, in_text_citations


# Example usage
if __name__ == "__main__":
    topic = "Digital Health Management"
    research_output, in_text_citations = generate_references_and_citations(topic)

    print(research_output)  # This will show the combined research and APA references
    print("\nIn-text citations for the research paper:")
    print("\n".join(in_text_citations))  # This will show the in-text citations for the topic