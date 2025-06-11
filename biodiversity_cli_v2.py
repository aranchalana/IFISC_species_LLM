#!/usr/bin/env python3
"""
Biodiversity Research Pipeline - Command Line Version
Comprehensive tool for species literature search and AI-powered data extraction

This application:
- Searches multiple databases (PubMed, CrossRef, bioRxiv, arXiv, Scopus)
- Extracts species data using Claude API or local Ollama
- Outputs results to CSV files for further analysis

Author: Biodiversity Research Tool
Version: 2.0 (Command Line)
Date: 2024
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import re
import xml.etree.ElementTree as ET
import csv
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings

warnings.filterwarnings('ignore')

class OllamaExtractor:
    """Species data extractor using local Ollama"""
    
    def __init__(self, model_name="phi3:mini", base_url="http://localhost:11434", use_streaming=False):
        self.model_name = model_name
        self.base_url = base_url
        self.use_streaming = use_streaming
        
    def test_connection(self):
        """Test if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def extract_species_data(self, papers: List[Dict], max_papers: int = None) -> List[Dict]:
        """Extract species data using local Ollama with aggressive timeout handling"""
        if max_papers:
            papers = papers[:max_papers]
        
        print(f"Extracting species data from {len(papers)} papers using Ollama ({self.model_name})")
        
        all_species_data = []
        
        for i, paper in enumerate(papers):
            try:
                print(f"Processing paper {i+1}/{len(papers)}: {paper.get('title', 'Unknown')[:50]}...")
                
                # Create minimal paper text to avoid timeouts
                title = paper.get('title', '')[:300]  # Limit title length
                abstract = paper.get('abstract', '')[:800]  # Limit abstract length
                
                if not title and not abstract:
                    print("  Skipping - no title or abstract")
                    continue
                
                # Much shorter, simpler prompt
                prompt = f"""Extract species names from this paper. Return only JSON array.

Title: {title}
Abstract: {abstract}

Find species names in "Genus species" format. Return as:
[{{"species":"Genus species","location":"study location","study_type":"Lab or Field"}}]

JSON:"""

                # Call Ollama API with aggressive retry and timeout handling
                success = False
                for attempt in range(2):  # Reduced retries
                    try:
                        print(f"  Attempt {attempt + 1}/2...")
                        response = requests.post(
                            f"{self.base_url}/api/generate",
                            json={
                                "model": self.model_name,
                                "prompt": prompt,
                                "stream": False,
                                "options": {
                                    "temperature": 0.0,  # More deterministic
                                    "top_p": 0.8,
                                    "num_ctx": 1024,  # Very small context
                                    "num_predict": 300,  # Short response
                                    "stop": ["\n\n", "```"]  # Stop early
                                }
                            },
                            timeout=180  # 3 minutes max
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            generated_text = result.get('response', '')
                            print(f"  Response received ({len(generated_text)} chars)")
                            
                            # Parse JSON from response
                            species_data = self.parse_llm_response(generated_text, paper)
                            all_species_data.extend(species_data)
                            
                            if species_data:
                                print(f"  Found {len(species_data)} species")
                            else:
                                print(f"  No species extracted")
                            
                            success = True
                            break
                        else:
                            print(f"  API error: {response.status_code}")
                            
                    except requests.exceptions.Timeout:
                        print(f"  Timeout on attempt {attempt + 1}")
                        if attempt == 0:  # Only retry once
                            print("  Retrying with even shorter text...")
                            # Try with even shorter text
                            prompt = f"Extract species names from: {title[:200]}. Return JSON: []"
                        else:
                            print("  Skipping paper due to timeout")
                            break
                    except Exception as e:
                        print(f"  Error: {e}")
                        break
                
                if not success:
                    print("  Failed to process paper, continuing...")
                
                # Minimal delay
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  Error processing paper: {e}")
                continue
        
        print(f"Ollama extraction complete! Found {len(all_species_data)} species entries")
        return all_species_data
    
    def parse_llm_response(self, response_text: str, paper: Dict) -> List[Dict]:
        """Parse LLM response and extract JSON"""
        try:
            # Find JSON in response
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if not json_match:
                # Try to find single object
                json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                if json_match:
                    json_text = '[' + json_match.group(0) + ']'
                else:
                    return []
            else:
                json_text = json_match.group(0)
            
            # Clean up common JSON issues
            json_text = re.sub(r',\s*}', '}', json_text)  # Remove trailing commas
            json_text = re.sub(r',\s*]', ']', json_text)  # Remove trailing commas
            
            result = json.loads(json_text)
            
            if isinstance(result, dict):
                result = [result]
            
            # Process results
            processed_data = []
            for item in result:
                if isinstance(item, dict):
                    clean_item = {
                        'query_species': paper.get('title', 'Unknown')[:50],
                        'paper_link': paper.get('doi', paper.get('url', 'Unknown')),
                        'species': str(item.get('species', 'Unknown')).strip(),
                        'number': str(item.get('number', 'unknown')).strip(),
                        'study_type': str(item.get('study_type', 'Unknown')).strip(),
                        'location': str(item.get('location', 'Unknown')).strip(),
                        'doi': paper.get('doi', ''),
                        'paper_title': paper.get('title', 'Unknown')
                    }
                    processed_data.append(clean_item)
            
            return processed_data
        
        except json.JSONDecodeError:
            # Try to extract individual fields if JSON parsing fails
            return self.fallback_parsing(response_text, paper)
        except Exception:
            return []
    
    def fallback_parsing(self, response_text: str, paper: Dict) -> List[Dict]:
        """Fallback parsing when JSON fails"""
        try:
            # Look for species names in the response
            species_patterns = [
                r'species["\s]*:["\s]*([A-Z][a-z]+\s+[a-z]+)',
                r'"([A-Z][a-z]+\s+[a-z]+)"',
                r'([A-Z][a-z]+\s+[a-z]+)'
            ]
            
            found_species = []
            for pattern in species_patterns:
                matches = re.findall(pattern, response_text)
                found_species.extend(matches)
            
            # Create basic entries for found species
            processed_data = []
            for species in found_species[:3]:  # Limit to 3 to avoid noise
                clean_item = {
                    'query_species': paper.get('title', 'Unknown')[:50],
                    'paper_link': paper.get('doi', paper.get('url', 'Unknown')),
                    'species': species.strip(),
                    'number': 'unknown',
                    'study_type': 'Unknown',
                    'location': 'Unknown',
                    'doi': paper.get('doi', ''),
                    'paper_title': paper.get('title', 'Unknown')
                }
                processed_data.append(clean_item)
            
            return processed_data
        except:
            return []

class BiodiversityPipeline:
    """Main pipeline class that combines all functionality"""
    
    def __init__(self):
        self.search_results = []
        self.species_data = []
        
    def clean_text_for_csv(self, text):
        """Clean text to prevent CSV parsing issues"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text).strip()
        text = re.sub(r'[\r\n]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('\x00', '')
        
        return text.strip()
    
    def search_pubmed(self, species: str, max_results: int = 25, start_year: int = 2015, end_year: int = 2025) -> List[Dict]:
        """Search PubMed database"""
        print(f"Searching PubMed for '{species}'...")
        
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        query = f'("{species}"[Title/Abstract]) AND ("{start_year}"[PDAT] : "{end_year}"[PDAT])'
        
        try:
            # Search for PMIDs
            search_response = requests.get(f"{base_url}/esearch.fcgi", params={
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'sort': 'relevance'
            }, timeout=30)
            search_response.raise_for_status()
            
            pmids = search_response.json().get('esearchresult', {}).get('idlist', [])
            if not pmids:
                print("  No PubMed results found")
                return []
            
            print(f"  Found {len(pmids)} PubMed results")
            
            # Fetch details
            results = []
            batch_size = 10
            
            for i in range(0, len(pmids), batch_size):
                batch_pmids = pmids[i:i+batch_size]
                
                fetch_response = requests.get(f"{base_url}/efetch.fcgi", params={
                    'db': 'pubmed',
                    'id': ','.join(batch_pmids),
                    'retmode': 'xml',
                    'rettype': 'abstract'
                }, timeout=30)
                fetch_response.raise_for_status()
                
                root = ET.fromstring(fetch_response.content)
                
                for article in root.findall('.//PubmedArticle'):
                    paper_data = self.parse_pubmed_article(article)
                    if paper_data:
                        results.append(paper_data)
                
                print(f"  Progress: {min(i + batch_size, len(pmids))}/{len(pmids)} papers processed")
                time.sleep(0.5)
            
            print(f"  Successfully parsed {len(results)} PubMed papers")
            return results
            
        except Exception as e:
            print(f"Error searching PubMed: {e}")
            return []
    
    def parse_pubmed_article(self, article_element) -> Optional[Dict]:
        """Parse PubMed article XML"""
        try:
            # Extract PMID
            pmid_elem = article_element.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else ""
            
            # Extract title
            title_elem = article_element.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""
            
            # Extract authors
            authors = []
            for author in article_element.findall('.//Author'):
                lastname = author.find('LastName')
                forename = author.find('ForeName')
                if lastname is not None and forename is not None:
                    authors.append(f"{lastname.text}, {forename.text}")
                elif lastname is not None:
                    authors.append(lastname.text)
            authors_str = "; ".join(authors)
            
            # Extract journal
            journal_elem = article_element.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Extract year
            year_elem = article_element.find('.//PubDate/Year')
            year = year_elem.text if year_elem is not None else ""
            
            # Extract abstract
            abstract_parts = []
            for abstract_text in article_element.findall('.//AbstractText'):
                if abstract_text.text:
                    abstract_parts.append(abstract_text.text)
            abstract = " ".join(abstract_parts)
            
            # Extract DOI
            doi = ""
            for article_id in article_element.findall('.//ArticleId'):
                if article_id.get('IdType') == 'doi':
                    doi = article_id.text
                    break
            
            return {
                'authors': authors_str,
                'journal': journal,
                'year': year,
                'abstract': abstract,
                'doi': doi,
                'pmid': pmid,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                'database': 'PubMed',
                'title': title
            }
            
        except Exception as e:
            print(f"Error parsing PubMed article: {e}")
            return None
    
    def search_crossref(self, species: str, max_results: int = 25, start_year: int = 2015, end_year: int = 2025) -> List[Dict]:
        """Search CrossRef database"""
        print(f"Searching CrossRef for '{species}'...")
        
        try:
            response = requests.get("https://api.crossref.org/works", params={
                'query': species,
                'rows': max_results,
                'filter': f'from-pub-date:{start_year},until-pub-date:{end_year}',
                'sort': 'relevance',
                'select': 'DOI,title,author,published-print,published-online,container-title,abstract,URL'
            }, headers={
                'User-Agent': 'Academic Research Tool (mailto:researcher@example.com)'
            }, timeout=30)
            response.raise_for_status()
            
            items = response.json().get('message', {}).get('items', [])
            if not items:
                print("  No CrossRef results found")
                return []
            
            print(f"  Found {len(items)} CrossRef results")
            
            results = []
            for item in items:
                paper_data = self.parse_crossref_item(item)
                if paper_data:
                    results.append(paper_data)
            
            print(f"  Successfully parsed {len(results)} CrossRef papers")
            return results
            
        except Exception as e:
            print(f"Error searching CrossRef: {e}")
            return []
    
    def parse_crossref_item(self, item: Dict) -> Optional[Dict]:
        """Parse CrossRef item"""
        try:
            title_list = item.get('title', [])
            title = title_list[0] if title_list else ""
            
            authors = []
            for author in item.get('author', []):
                given = author.get('given', '')
                family = author.get('family', '')
                if family:
                    if given:
                        authors.append(f"{family}, {given}")
                    else:
                        authors.append(family)
            authors_str = "; ".join(authors)
            
            container_title = item.get('container-title', [])
            journal = container_title[0] if container_title else ""
            
            year = ""
            pub_date = item.get('published-print') or item.get('published-online')
            if pub_date and 'date-parts' in pub_date:
                date_parts = pub_date['date-parts'][0]
                if date_parts:
                    year = str(date_parts[0])
            
            return {
                'authors': authors_str,
                'journal': journal,
                'year': year,
                'abstract': item.get('abstract', ''),
                'doi': item.get('DOI', ''),
                'pmid': '',
                'url': item.get('URL', ''),
                'database': 'CrossRef',
                'title': title
            }
            
        except Exception as e:
            print(f"Error parsing CrossRef item: {e}")
            return None
    
    def search_biorxiv(self, species: str, max_results: int = 25, start_year: int = 2015, end_year: int = 2025) -> List[Dict]:
        """Search bioRxiv database"""
        print(f"Searching bioRxiv for '{species}'...")
        
        try:
            start_date = f"{start_year}-01-01"
            end_date = f"{end_year}-12-31"
            
            response = requests.get(f"https://api.biorxiv.org/details/biorxiv/{start_date}/{end_date}", timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if 'collection' not in data:
                print("  No bioRxiv results found")
                return []
            
            # Filter by species name
            all_papers = data['collection']
            filtered_papers = []
            species_lower = species.lower()
            
            for paper in all_papers:
                title = paper.get('title', '').lower()
                abstract = paper.get('abstract', '').lower()
                
                if species_lower in title or species_lower in abstract:
                    filtered_papers.append(paper)
                    
                if len(filtered_papers) >= max_results:
                    break
            
            if not filtered_papers:
                print("  No relevant bioRxiv results found")
                return []
            
            print(f"  Found {len(filtered_papers)} relevant bioRxiv results")
            
            results = []
            for paper in filtered_papers:
                results.append({
                    'authors': paper.get('authors', ''),
                    'journal': 'bioRxiv (preprint)',
                    'year': paper.get('date', '')[:4] if paper.get('date') else '',
                    'abstract': paper.get('abstract', ''),
                    'doi': paper.get('doi', ''),
                    'pmid': '',
                    'url': f"https://www.biorxiv.org/content/{paper.get('doi', '')}v1",
                    'database': 'bioRxiv',
                    'title': paper.get('title', '')
                })
            
            print(f"  Successfully parsed {len(results)} bioRxiv papers")
            return results
            
        except Exception as e:
            print(f"Error searching bioRxiv: {e}")
            return []
    
    def search_arxiv(self, species: str, max_results: int = 25, start_year: int = 2015, end_year: int = 2025) -> List[Dict]:
        """Search arXiv database"""
        print(f"Searching arXiv for '{species}'...")
        
        try:
            response = requests.get("http://export.arxiv.org/api/query", params={
                'search_query': f'all:"{species}"',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', namespace)
            
            if not entries:
                print("  No arXiv results found")
                return []
            
            print(f"  Found {len(entries)} arXiv results")
            
            results = []
            for entry in entries:
                paper_data = self.parse_arxiv_entry(entry, namespace, start_year, end_year)
                if paper_data:
                    results.append(paper_data)
            
            print(f"  Successfully parsed {len(results)} arXiv papers")
            return results
            
        except Exception as e:
            print(f"Error searching arXiv: {e}")
            return []
    
    def parse_arxiv_entry(self, entry, namespace: Dict, start_year: int, end_year: int) -> Optional[Dict]:
        """Parse arXiv entry"""
        try:
            title_elem = entry.find('atom:title', namespace)
            title = title_elem.text.strip() if title_elem is not None else ""
            
            authors = []
            for author in entry.findall('atom:author', namespace):
                name_elem = author.find('atom:name', namespace)
                if name_elem is not None:
                    authors.append(name_elem.text)
            authors_str = "; ".join(authors)
            
            published_elem = entry.find('atom:published', namespace)
            published = published_elem.text if published_elem is not None else ""
            year = published[:4] if len(published) >= 4 else ""
            
            if year and (int(year) < start_year or int(year) > end_year):
                return None
            
            summary_elem = entry.find('atom:summary', namespace)
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            
            id_elem = entry.find('atom:id', namespace)
            arxiv_url = id_elem.text if id_elem is not None else ""
            
            return {
                'authors': authors_str,
                'journal': 'arXiv (preprint)',
                'year': year,
                'abstract': abstract,
                'doi': '',
                'pmid': '',
                'url': arxiv_url,
                'database': 'arXiv',
                'title': title
            }
            
        except Exception as e:
            print(f"Error parsing arXiv entry: {e}")
            return None
    
    def search_scopus(self, species: str, api_key: str, max_results: int = 25, start_year: int = 2015, end_year: int = 2025, inst_token: str = None) -> List[Dict]:
        """Search Scopus database"""
        if not api_key:
            print("  Skipping Scopus search (no API key)")
            return []
        
        print(f"Searching Scopus for '{species}'...")
        
        try:
            headers = {
                'X-ELS-APIKey': api_key,
                'Accept': 'application/json'
            }
            
            if inst_token:
                headers['X-ELS-Insttoken'] = inst_token
            
            response = requests.get("https://api.elsevier.com/content/search/scopus", headers=headers, params={
                'query': f'TITLE-ABS-KEY("{species}") AND PUBYEAR > {start_year-1} AND PUBYEAR < {end_year+1}',
                'count': max_results,
                'sort': 'relevancy',
                'field': 'dc:title,dc:creator,prism:publicationName,prism:coverDate,dc:description,prism:doi,dc:identifier,prism:url'
            }, timeout=30)
            response.raise_for_status()
            
            entries = response.json().get('search-results', {}).get('entry', [])
            if not entries:
                print("  No Scopus results found")
                return []
            
            print(f"  Found {len(entries)} Scopus results")
            
            results = []
            for entry in entries:
                results.append({
                    'authors': entry.get('dc:creator', ''),
                    'journal': entry.get('prism:publicationName', ''),
                    'year': entry.get('prism:coverDate', '')[:4] if entry.get('prism:coverDate') else '',
                    'abstract': entry.get('dc:description', ''),
                    'doi': entry.get('prism:doi', ''),
                    'pmid': '',
                    'url': entry.get('prism:url', ''),
                    'database': 'Scopus',
                    'title': entry.get('dc:title', '')
                })
            
            print(f"  Successfully parsed {len(results)} Scopus papers")
            return results
            
        except Exception as e:
            print(f"Error searching Scopus: {e}")
            return []
    
    def remove_duplicates(self, papers: List[Dict]) -> List[Dict]:
        """Remove duplicate papers"""
        if not papers:
            return []
        
        unique_papers = []
        seen_dois = set()
        seen_titles = set()
        
        for paper in papers:
            doi = paper.get('doi', '').strip()
            title = paper.get('title', '').strip().lower()
            
            # Check DOI duplicates
            if doi and doi in seen_dois:
                continue
            
            # Check title similarity
            title_words = set(title.split())
            is_duplicate = False
            
            for seen_title in seen_titles:
                seen_words = set(seen_title.split())
                if title_words and seen_words:
                    overlap = len(title_words & seen_words) / len(title_words | seen_words)
                    if overlap > 0.8:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_papers.append(paper)
                if doi:
                    seen_dois.add(doi)
                if title:
                    seen_titles.add(title)
        
        return unique_papers
    
    def search_all_databases(self, species: str, start_year: int, end_year: int, max_results: int, 
                           scopus_key: str = None, scopus_token: str = None) -> List[Dict]:
        """Search all available databases"""
        print(f"Searching all databases for: {species}")
        
        all_papers = []
        
        # Define search functions
        search_functions = [
            ("PubMed", lambda: self.search_pubmed(species, max_results, start_year, end_year)),
            ("CrossRef", lambda: self.search_crossref(species, max_results, start_year, end_year)),
            ("bioRxiv", lambda: self.search_biorxiv(species, max_results, start_year, end_year)),
            ("arXiv", lambda: self.search_arxiv(species, max_results, start_year, end_year)),
        ]
        
        if scopus_key:
            search_functions.append(("Scopus", lambda: self.search_scopus(species, scopus_key, max_results, start_year, end_year, scopus_token)))
        
        # Execute searches
        database_results = {}
        
        for i, (db_name, search_func) in enumerate(search_functions):
            print(f"\n--- {db_name} ({i+1}/{len(search_functions)}) ---")
            try:
                results = search_func()
                all_papers.extend(results)
                database_results[db_name] = len(results)
                print(f"{db_name}: {len(results)} papers found")
            except Exception as e:
                print(f"{db_name}: Error - {e}")
                database_results[db_name] = 0
            
            if i < len(search_functions) - 1:
                time.sleep(1)
        
        # Remove duplicates
        print(f"\nTotal papers before deduplication: {len(all_papers)}")
        unique_papers = self.remove_duplicates(all_papers)
        print(f"Unique papers after deduplication: {len(unique_papers)}")
        
        # Show database breakdown
        print("\nPapers by database:")
        for db, count in database_results.items():
            print(f"  {db}: {count}")
        
        return unique_papers
    
    def extract_species_data_with_claude(self, papers: List[Dict], claude_api_key: str, max_papers: int = None) -> List[Dict]:
        """Extract species data using Claude API"""
        if not claude_api_key:
            print("Claude API key is required for species extraction")
            return []
        
        if max_papers:
            papers = papers[:max_papers]
        
        print(f"Extracting species data from {len(papers)} papers using Claude API")
        
        all_species_data = []
        
        headers = {
            "x-api-key": claude_api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        for i, paper in enumerate(papers):
            try:
                print(f"Processing paper {i+1}/{len(papers)}: {paper.get('title', 'Unknown')[:50]}...")
                
                # Create paper text
                text_parts = []
                
                if paper.get('title'):
                    text_parts.append(f"Title: {paper['title']}")
                
                if paper.get('abstract'):
                    text_parts.append(f"Abstract: {paper['abstract']}")
                
                for field in ['authors', 'journal', 'year']:
                    if paper.get(field):
                        text_parts.append(f"{field.title()}: {paper[field]}")
                
                if not text_parts:
                    continue
                
                paper_text = "\n\n".join(text_parts)
                
                # Claude API prompt
                prompt = f"""
                Extract species information from this research paper. Return ONLY a JSON array of objects.

                For each species mentioned in the study (not just examples or background), extract:
                - species: scientific name (Genus species format)
                - number: specimen count or "number not specified"
                - study_type: "Laboratory", "Field", or "Field+Laboratory"
                - location: study location/site

                Return format (use simple strings only, no nested objects):
                [
                  {{
                    "species": "Genus species",
                    "number": "count or number not specified",
                    "study_type": "Laboratory/Field/Field+Laboratory",
                    "location": "location description"
                  }}
                ]

                Paper: {paper.get('title', 'Unknown')}
                DOI: {paper.get('doi', 'Unknown')}

                Text to analyze:
                {paper_text[:50000]}
                """
                
                payload = {
                    "model": "claude-3-haiku-20240307",
                    "max_tokens": 1500,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.0
                }
                
                # Make API request with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = requests.post(
                            "https://api.anthropic.com/v1/messages",
                            headers=headers,
                            json=payload, 
                            timeout=60
                        )
                        
                        if response.status_code == 429:
                            wait_time = min(2 ** attempt, 60)
                            print(f"Rate limit hit. Waiting {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                            
                        if response.status_code != 200:
                            raise Exception(f"API request failed: {response.text}")
                            
                        break
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = min(2 ** attempt, 60)
                            print(f"Error: {e}. Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            raise
                else:
                    raise Exception("Max retries exceeded")
                
                # Parse Claude response
                response_data = response.json()
                claude_response = response_data["content"][0]["text"]
                
                # Clean and parse JSON
                try:
                    # Remove markdown formatting
                    claude_response = re.sub(r'```(?:json)?\n', '', claude_response)
                    claude_response = re.sub(r'\n```', '', claude_response)
                    
                    json_match = re.search(r'(\[.*\]|\{.*\})', claude_response, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(1)
                    else:
                        json_text = claude_response
                    
                    result = json.loads(json_text)
                    
                    if isinstance(result, dict):
                        result = [result]
                    
                    # Process results
                    for item in result:
                        if isinstance(item, dict):
                            clean_item = {
                                'query_species': paper.get('title', 'Unknown')[:50],
                                'paper_link': paper.get('doi', paper.get('url', 'Unknown')),
                                'species': str(item.get('species', 'Unknown')).strip(),
                                'number': str(item.get('number', 'unknown')).strip(),
                                'study_type': str(item.get('study_type', 'Unknown')).strip(),
                                'location': str(item.get('location', 'Unknown')).strip(),
                                'doi': paper.get('doi', ''),
                                'paper_title': paper.get('title', 'Unknown')
                            }
                            all_species_data.append(clean_item)
                    
                    if result:
                        print(f"Found {len(result)} species in this paper")
                    
                except json.JSONDecodeError:
                    print(f"Could not parse Claude response for this paper")
                    continue
                
                # Rate limiting
                if i < len(papers) - 1:
                    time.sleep(5)
                    
            except Exception as e:
                print(f"Error processing paper: {e}")
                continue
        
        print(f"Species extraction complete! Found {len(all_species_data)} species entries")
        return all_species_data


def save_to_csv(data: List[Dict], filename: str):
    """Save data to CSV file"""
    if not data:
        print(f"No data to save to {filename}")
        return
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Biodiversity Research Pipeline - Command Line Version')
    
    # Required arguments
    parser.add_argument('species', nargs='+', help='Species name(s) to search for')
    
    # AI Backend selection
    parser.add_argument('--ai-backend', choices=['claude', 'ollama'], default='claude',
                        help='AI backend to use for species extraction (default: claude)')
    
    # Claude API arguments
    parser.add_argument('--claude-api-key', help='Claude API key (required if using Claude backend)')
    
    # Ollama arguments
    parser.add_argument('--ollama-model', default='llama3.1:8b',
                        help='Ollama model to use (default: llama3.1:8b)')
    parser.add_argument('--ollama-url', default='http://localhost:11434',
                        help='Ollama base URL (default: http://localhost:11434)')
    
    # Database arguments
    parser.add_argument('--scopus-api-key', help='Scopus API key (optional)')
    parser.add_argument('--scopus-token', help='Scopus institutional token (optional)')
    
    # Search parameters
    parser.add_argument('--start-year', type=int, default=2015,
                        help='Start year for search (default: 2015)')
    parser.add_argument('--end-year', type=int, default=2025,
                        help='End year for search (default: 2025)')
    parser.add_argument('--max-results', type=int, default=25,
                        help='Maximum papers per database (default: 25)')
    parser.add_argument('--max-extract', type=int, default=50,
                        help='Maximum papers to extract data from (default: 50)')
    
    # Output arguments
    parser.add_argument('--output-dir', default='.',
                        help='Output directory for CSV files (default: current directory)')
    parser.add_argument('--prefix', default='biodiversity',
                        help='Prefix for output files (default: biodiversity)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.ai_backend == 'claude' and not args.claude_api_key:
        print("Error: Claude API key is required when using Claude backend")
        print("Use --claude-api-key or switch to --ai-backend ollama")
        return
    
    if args.ai_backend == 'ollama':
        # Test Ollama connection
        extractor = OllamaExtractor(args.ollama_model, args.ollama_url)
        if not extractor.test_connection():
            print(f"Error: Cannot connect to Ollama at {args.ollama_url}")
            print("Make sure Ollama is running with: ollama serve")
            return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Initialize pipeline
    pipeline = BiodiversityPipeline()
    
    print("Biodiversity Research Pipeline - Command Line Version")
    print("=" * 50)
    print(f"AI Backend: {args.ai_backend}")
    print(f"Species: {', '.join(args.species)}")
    print(f"Year range: {args.start_year}-{args.end_year}")
    print(f"Max papers per database: {args.max_results}")
    print(f"Max papers for extraction: {args.max_extract}")
    print("=" * 50)
    
    all_search_results = []
    all_species_data = []
    
    # Process each species
    for i, species in enumerate(args.species):
        print(f"\nProcessing species {i+1}/{len(args.species)}: {species}")
        print("-" * 40)
        
        # Search databases
        search_results = pipeline.search_all_databases(
            species, args.start_year, args.end_year, args.max_results,
            args.scopus_api_key, args.scopus_token
        )
        
        if search_results:
            all_search_results.extend(search_results)
            
            # Extract species data using chosen backend
            if args.ai_backend == 'claude':
                species_data = pipeline.extract_species_data_with_claude(
                    search_results, args.claude_api_key, args.max_extract
                )
            else:  # ollama
                extractor = OllamaExtractor(args.ollama_model, args.ollama_url)
                species_data = extractor.extract_species_data(search_results, args.max_extract)
            
            if species_data:
                all_species_data.extend(species_data)
        
        print(f"Completed {species}: {len(search_results)} papers, {len(species_data) if search_results else 0} species records")
    
    # Save results
    print("\n" + "=" * 50)
    print("SAVING RESULTS")
    print("=" * 50)
    
    if all_search_results:
        papers_file = output_dir / f"{args.prefix}_papers_{timestamp}.csv"
        save_to_csv(all_search_results, papers_file)
    
    if all_species_data:
        species_file = output_dir / f"{args.prefix}_species_{timestamp}.csv"
        save_to_csv(all_species_data, species_file)
    
    # Print summary
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total papers found: {len(all_search_results)}")
    print(f"Total species records: {len(all_species_data)}")
    
    if all_species_data:
        species_df = pd.DataFrame(all_species_data)
        unique_species = species_df['species'].nunique()
        unique_locations = species_df['location'].nunique()
        print(f"Unique species: {unique_species}")
        print(f"Unique locations: {unique_locations}")
        
        # Top species
        print("\nTop 5 most studied species:")
        top_species = species_df['species'].value_counts().head(5)
        for i, (species, count) in enumerate(top_species.items(), 1):
            print(f"  {i}. {species}: {count} records")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
