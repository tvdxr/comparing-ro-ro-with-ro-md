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
DEFAULT_PAGES = 5

# Robust Headers
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
}

def save_article_simple(site_name, index, url, title, content):
    """Saves article: site_1.json"""
    if not content or len(content) < 150:
        return False

    site_dir = BASE_OUTPUT_DIR / site_name
    site_dir.mkdir(parents=True, exist_ok=True)
    
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
    Robust content extraction based on text density.
    """
    for tag in soup(['script', 'style', 'noscript', 'iframe', 'header', 'footer', 'nav', 'aside', 'form', 'div.comments', 'div.related', 'div.share']):
        tag.decompose()

    # Priority 1: Known Content Classes
    common_classes = [
        'entry-content', 'td-post-content', 'post-content', 'news-text', 
        'article-body', 'node-content', 'field-name-body', 'details-content',
        'text-content', 'news_text', 'article-text', 'full-text', 'art-body'
    ]
    for cls in common_classes:
        div = soup.find('div', class_=cls)
        if div:
            text = div.get_text(separator="\n").strip()
            if len(text) > 150: return text

    # Priority 2: Text Density
    best_tag = None
    max_len = 0
    
    for tag in soup.find_all(['div', 'article', 'section']):
        paras = tag.find_all('p', recursive=False)
        if not paras: paras = tag.find_all('p')
        
        current_len = sum(len(p.get_text()) for p in paras)
        
        if current_len > max_len:
            max_len = current_len
            best_tag = tag

    if best_tag and max_len > 150:
        return best_tag.get_text(separator="\n").strip()

    return None

def is_romanian(text):
    """Simple check to filter out English/Russian posts"""
    common_words = [' și ', ' de ', ' nu ', ' la ', ' care ', ' este ', ' sunt ', ' eu ', ' tu ', ' el ', ' pe ', ' cu ']
    hit_count = sum(1 for word in common_words if word in text.lower())
    return hit_count >= 2

def crawl_reddit_moldova(limit=200):
    print(f"\n--- Crawling Reddit (r/moldova) ---")
    
    # Reddit pagination uses 'after' token
    after = None
    saved_count = 0
    pages_to_fetch = int(limit / 100) + 1
    
    for _ in range(pages_to_fetch):
        url = f"https://www.reddit.com/r/moldova/new.json?limit=100"
        if after:
            url += f"&after={after}"
            
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                print(f"   [!] Reddit blocked the request: {resp.status_code}")
                break

            data = resp.json()
            posts = data.get('data', {}).get('children', [])
            after = data.get('data', {}).get('after')
            
            if not posts:
                break

            for post in posts:
                post_data = post['data']
                title = post_data.get('title', '')
                selftext = post_data.get('selftext', '')
                post_url = "https://reddit.com" + post_data.get('permalink', '')
                
                full_text = f"{title}\n\n{selftext}"
                
                # Simple deduplication handled by file indexing
                if len(selftext) > 50 and is_romanian(full_text):
                    saved_count += 1
                    if save_article_simple("reddit_moldova", saved_count, post_url, title, selftext):
                        print(f"      [{saved_count}] Saved: {title[:30]}...")
            
            if not after:
                break
            time.sleep(2) # Polite delay between reddit pages

        except Exception as e:
            print(f"   Error crawling Reddit: {e}")
            break
            
    print(f"   Finished Reddit. Saved {saved_count} informal posts.")

def crawl_site_pages(site_name, base_url, url_template, link_regex, max_pages=DEFAULT_PAGES):
    print(f"\n--- Crawling {site_name} ---")
    visited_urls = set()
    global_index = 1
    
    for page in range(1, max_pages + 1):
        # Build URL
        if "{}" in url_template:
            url = url_template.format(page)
        else:
            url = f"{base_url}{url_template}".format(page)

        print(f"Scanning Page {page}: {url}...")
        
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15, verify=False)
            if resp.status_code != 200:
                print(f"   [!] Status {resp.status_code} - Skipping")
                continue
            
            # --- FIX: Force Encoding for sites like Realitatea ---
            if resp.encoding is None or resp.encoding == 'ISO-8859-1':
                resp.encoding = resp.apparent_encoding

            soup = BeautifulSoup(resp.text, 'html.parser')
            all_links = soup.find_all('a', href=True)
            candidates = []

            for a in all_links:
                href = a['href']
                
                # Normalize URL
                if href.startswith('/'):
                    parsed = url.split('/')
                    if len(parsed) >= 3:
                        domain = f"{parsed[0]}//{parsed[2]}"
                        href = domain + href
                    else:
                        continue
                elif not href.startswith('http'):
                    continue

                # Broad Regex Filter
                if re.search(link_regex, href):
                    # Robust junk filter
                    if not any(x in href for x in ['/category/', '/page/', '/tag/', '/search/', '#', '.jpg', 'login', 'contact', 'publicitate', 'rss', 'facebook', 'twitter']):
                        candidates.append(href)

            candidates = list(set(candidates))
            print(f"   Found {len(candidates)} potential links.")

            for link in candidates:
                if link not in visited_urls:
                    visited_urls.add(link)
                    
                    try:
                        art_resp = requests.get(link, headers=HEADERS, timeout=10, verify=False)
                        if art_resp.status_code == 200:
                            # --- FIX: Force Encoding here as well ---
                            if art_resp.encoding is None or art_resp.encoding == 'ISO-8859-1':
                                art_resp.encoding = art_resp.apparent_encoding

                            art_soup = BeautifulSoup(art_resp.text, 'html.parser')
                            
                            h1 = art_soup.find('h1')
                            title = h1.get_text().strip() if h1 else "No Title"
                            content = extract_content_heuristic(art_soup)
                            
                            if content:
                                if save_article_simple(site_name, global_index, link, title, content):
                                    print(f"      [{global_index}] Saved: {title[:30]}...")
                                    global_index += 1
                                    time.sleep(0.2)
                    except Exception:
                        pass
        except Exception as e:
            print(f"   Error scanning page: {e}")

if __name__ == "__main__":
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- 1. COMPLETED SITES (Commented out) ---
    # crawl_reddit_moldova(limit=300)
    
    # sites_done = [
    #     {
    #         "name": "zugo.md",
    #         "base": "https://zugo.md/category/actualitate/",
    #         "template": "https://zugo.md/category/actualitate/page/{}/",
    #         "regex": r"zugo\.md\/[\w-]+"
    #     },
    #     {
    #         "name": "diez.md",
    #         "base": "https://diez.md/category/social/",
    #         "template": "https://diez.md/category/social/page/{}/",
    #         "regex": r"diez\.md\/[\w-]+"
    #     },
    #     {
    #         "name": "moldova.org",
    #         "base": "https://www.moldova.org/category/social/",
    #         "template": "https://www.moldova.org/category/social/page/{}/",
    #         "regex": r"moldova\.org\/[\w-]+"
    #     },
    #     {
    #         "name": "zdg.md",
    #         "base": "https://www.zdg.md/category/stiri/social/",
    #         "template": "https://www.zdg.md/category/stiri/social/page/{}/",
    #         "regex": r"zdg\.md\/[\w-]+",
    #         "pages": 15
    #     }
    # ]

    # --- 2. SITES TO RETRY / FIX ---
    sites_to_crawl = [
        {
            "name": "realitatea.md",
            # Added encoding fix in crawl_site_pages
            "base": "https://realitatea.md/category/societate/",
            "template": "https://realitatea.md/category/societate/page/{}/",
            "regex": r"realitatea\.md\/[\w-]+"
        },
        {
            "name": "agora.md",
            # Broadened regex: agora.md + any path
            "base": "https://agora.md/categorie/social",
            "template": "https://agora.md/categorie/social?page={}",
            "regex": r"agora\.md\/stiri\/.+" 
        },
        {
            "name": "unimedia.info",
            # Broadened regex: unimedia.info + any path
            "base": "https://unimedia.info/ro/category/social",
            "template": "https://unimedia.info/ro/category/social/page/{}",
            "regex": r"unimedia\.info\/ro\/(news|stiri)\/.+"
        },
        {
            "name": "deschide.md",
            # Broadened regex
            "base": "https://deschide.md/ro/stiri/social/",
            "template": "https://deschide.md/ro/stiri/social/page/{}/",
            "regex": r"deschide\.md\/ro\/stiri\/.+"
        },
        {
            "name": "stiri.md",
            # Broadened regex
            "base": "https://stiri.md/category/social/",
            "template": "https://stiri.md/category/social/page/{}/",
            "regex": r"stiri\.md\/article\/.+"
        },
        {
            "name": "shok.md",
            # Shok regex
            "base": "https://shok.md/monden/",
            "template": "https://shok.md/monden/page/{}/",
            "regex": r"shok\.md\/[\w-]+"
        }
    ]

    for s in sites_to_crawl:
        p_count = s.get("pages", DEFAULT_PAGES)
        crawl_site_pages(s["name"], s["base"], s["template"], s["regex"], max_pages=p_count)

    print(f"\nCrawling complete. Files saved in {BASE_OUTPUT_DIR}")
    