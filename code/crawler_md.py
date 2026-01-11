import requests
from bs4 import BeautifulSoup
import time
import json
import re
from pathlib import Path
import urllib3
import warnings

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- CONFIGURATION ---
BASE_OUTPUT_DIR = Path("data_cleaned/md_crawl_data")
PAGES_TO_CRAWL = 5

# Robust Headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Referer': 'https://www.google.com/',
}

def save_article_simple(site_name, index, url, title, content):
    """
    Saves article with simple index filename: site_1.json
    """
    if not content or len(content) < 150:
        return False

    # Create folder: data_cleaned/md_crawl_data/zugo.md/
    site_dir = BASE_OUTPUT_DIR / site_name
    site_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple filename: zugo.md_1.json
    filename = f"{site_name}_{index}.json"
    file_path = site_dir / filename

    data_entry = {
        "title": title,
        "content": content,
        "metadata": {
            "id": index,
            "source_url": url
        }
    }

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_entry, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving {filename}: {e}")
        return False

def extract_content_heuristic(soup):
    """
    Finds the text content by looking for the element with the most paragraphs.
    """
    # Remove junk
    for tag in soup(['script', 'style', 'noscript', 'iframe', 'header', 'footer', 'nav', 'aside', 'form', 'div.comments', 'div.related-posts']):
        tag.decompose()

    # Strategy 1: Known Content Classes
    common_classes = [
        'entry-content', 'td-post-content', 'post-content', 'news-text', 
        'article-body', 'node-content', 'field-name-body'
    ]
    for cls in common_classes:
        div = soup.find('div', class_=cls)
        if div:
            text = div.get_text(separator="\n").strip()
            if len(text) > 150: return text

    # Strategy 2: Text Density (The "Nuclear Option")
    # Find ANY tag that contains a lot of text inside <p> tags
    best_tag = None
    max_len = 0
    
    # Scan all divs and articles
    for tag in soup.find_all(['div', 'article', 'section']):
        # Calculate length of text in direct paragraphs
        paras = tag.find_all('p', recursive=False)
        if not paras: paras = tag.find_all('p') # check deeper if no direct p
        
        # quick sum
        current_len = sum(len(p.get_text()) for p in paras)
        
        if current_len > max_len:
            max_len = current_len
            best_tag = tag

    if best_tag and max_len > 150:
        return best_tag.get_text(separator="\n").strip()

    return None

def crawl_site_pages(site_name, base_url_template, link_regex):
    print(f"\n--- Crawling {site_name} ---")
    visited_urls = set()
    global_index = 1
    
    for page in range(1, PAGES_TO_CRAWL + 1):
        # Build URL
        if "{}" in base_url_template:
            url = base_url_template.format(page)
        else:
            url = f"{base_url_template}?page={page}"

        print(f"Scanning Page {page}: {url}...")
        
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15, verify=False)
            if resp.status_code != 200:
                print(f"   [!] Status {resp.status_code} - Skipping")
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')
            
            # Find ALL links
            all_links = soup.find_all('a', href=True)
            candidates = []

            for a in all_links:
                href = a['href']
                
                # Make absolute URL
                if href.startswith('/'):
                    # e.g. /article-name -> https://site.md/article-name
                    parsed = url.split('/')
                    domain = f"{parsed[0]}//{parsed[2]}"
                    href = domain + href
                elif not href.startswith('http'):
                    continue

                # Filter URLs
                if re.search(link_regex, href):
                    # Exclude junk
                    if not any(x in href for x in ['/category/', '/page/', '/tag/', '/search/', '#', '.jpg']):
                        candidates.append(href)

            # Deduplicate
            candidates = list(set(candidates))
            print(f"   Found {len(candidates)} potential links.")

            for link in candidates:
                if link not in visited_urls:
                    visited_urls.add(link)
                    
                    # Process Article
                    try:
                        art_resp = requests.get(link, headers=HEADERS, timeout=10, verify=False)
                        if art_resp.status_code == 200:
                            art_soup = BeautifulSoup(art_resp.text, 'html.parser')
                            
                            # Title
                            h1 = art_soup.find('h1')
                            title = h1.get_text().strip() if h1 else "No Title"
                            
                            # Content
                            content = extract_content_heuristic(art_soup)
                            
                            if content:
                                if save_article_simple(site_name, global_index, link, title, content):
                                    print(f"      [{global_index}] Saved: {title[:30]}...")
                                    global_index += 1
                                    time.sleep(0.2)
                    except Exception:
                        pass
                        
        except Exception as e:
            print(f"   Error: {e}")

if __name__ == "__main__":
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. ZUGO.MD 
    # Valid Categories: Actualitate, Lifestyle
    # Regex: zugo.md/ followed by anything except common path segments
    crawl_site_pages(
        "zugo.md", 
        "https://zugo.md/category/actualitate/page/{}/", 
        r"zugo\.md\/[\w-]+" 
    )

    # 2. EA.MD 
    # Valid Categories: Social (checked via browse tool), Lifestyle-si-cariera
    crawl_site_pages(
        "ea.md", 
        "https://ea.md/category/social/page/{}/", 
        r"ea\.md\/[\w-]+" 
    )

    # 3. PERFECTE.MD
    # Valid: /lifestyle?page=X
    # Regex: perfecte.md/ followed by anything. 
    crawl_site_pages(
        "perfecte.md", 
        "https://perfecte.md/lifestyle?page={}", 
        r"perfecte\.md\/[\w-]+" 
    )

    print(f"\nCrawling complete. Files saved in {BASE_OUTPUT_DIR}")