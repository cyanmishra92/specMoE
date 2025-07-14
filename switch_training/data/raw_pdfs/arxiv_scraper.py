#!/usr/bin/env python3
"""
ArXiv Paper Scraper with Configuration Files
Downloads papers and metadata from arXiv based on flexible configuration files
"""

import requests
import xml.etree.ElementTree as ET
import os
import time
import json
import yaml
from datetime import datetime
from urllib.parse import urlencode
import re
from typing import List, Dict, Set, Optional
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration files for different search scenarios"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._create_default_configs()
    
    def _create_default_configs(self):
        """Create default configuration files if they don't exist"""
        
        # 1. Computer Architecture + ML + DL Config
        comp_arch_ml_config = {
            "name": "Computer Architecture + Machine Learning + Deep Learning",
            "description": "Papers combining computer architecture with ML/DL research",
            "search_mode": "general",
            "max_papers": 150,
            "download_dir": "./papers",
            "target_authors": [],
            "target_organizations": [],
            "custom_queries": [
                'cat:cs.AR AND cat:cs.LG',
                'cat:cs.AR AND cat:cs.AI', 
                'cat:cs.DC AND cat:cs.LG',
                'cat:cs.PF AND cat:cs.LG',
                '"computer architecture" AND "machine learning"',
                '"computer architecture" AND "deep learning"',
                '"neural architecture" AND "hardware"',
                '"accelerator" AND "deep learning"',
                '"FPGA" AND "machine learning"',
                '"GPU" AND "neural network"',
                '"hardware acceleration" AND "AI"',
                '"systolic array" AND "deep learning"',
                '"neuromorphic" AND "architecture"',
                '"inference acceleration"',
                '"training acceleration"',
                '"edge computing" AND "machine learning"',
                '"TPU" AND "deep learning"',
                '"tensor processing unit"',
                '"neural processing unit"',
                '"AI accelerator"',
                '"dataflow architecture" AND "neural"',
                '"quantization" AND "hardware"',
                '"pruning" AND "acceleration"',
                '"sparse neural networks" AND "hardware"'
            ],
            "relevance_filters": {
                "require_both_topics": True,
                "architecture_keywords": [
                    "architecture", "hardware", "accelerator", "fpga", "gpu", "asic",
                    "systolic", "neuromorphic", "processor", "chip", "circuit",
                    "performance", "optimization", "parallel", "distributed",
                    "edge computing", "inference", "training acceleration", "tpu"
                ],
                "ml_keywords": [
                    "machine learning", "deep learning", "neural network", "ai",
                    "artificial intelligence", "model", "inference", "training",
                    "cnn", "rnn", "transformer", "attention", "embedding"
                ]
            }
        }
        
        # 2. Specific Authors Config
        authors_config = {
            "name": "Specific Authors in ML + Computer Architecture",
            "description": "Papers from leading researchers in ML and computer architecture",
            "search_mode": "authors",
            "max_papers": 100,
            "download_dir": "./papers",
            "target_authors": [
                "Song Han",           # MIT, efficient neural networks
                "Bill Dally",         # Stanford/NVIDIA, computer architecture  
                "Krste Asanovic",     # UC Berkeley, RISC-V, computer architecture
                "David Patterson",    # UC Berkeley, computer architecture
                "Vivienne Sze",       # MIT, energy-efficient computing
                "Joel Emer",          # MIT/NVIDIA, computer architecture
                "Mark Horowitz",      # Stanford, computer architecture
                "Kunle Olukotun",     # Stanford, parallel computing
                "Margaret Martonosi", # Princeton, computer architecture
                "Luis Ceze",          # University of Washington
                "Onur Mutlu",         # ETH Zurich, computer architecture
                "Babak Falsafi",      # EPFL, computer architecture
                "Gu-Yeon Wei",        # Harvard, computer architecture
                "Hadi Esmaeilzadeh",  # UC San Diego, computer architecture
                "Yann LeCun",         # Meta/NYU, deep learning
                "Geoffrey Hinton",    # Google/University of Toronto
                "Fei-Fei Li",        # Stanford, computer vision
                "Andrew Ng",          # Stanford, machine learning
                "Yoshua Bengio"       # University of Montreal, deep learning
            ],
            "target_organizations": [],
            "custom_queries": [],
            "relevance_filters": {
                "require_both_topics": False,
                "architecture_keywords": [],
                "ml_keywords": []
            }
        }
        
        # 3. Universities and Institutions Config
        institutions_config = {
            "name": "Leading Universities and Research Institutions",
            "description": "Papers from top universities and research labs in ML and computer architecture",
            "search_mode": "organizations",
            "max_papers": 200,
            "download_dir": "./papers",
            "target_authors": [],
            "target_organizations": [
                # Universities
                "MIT",
                "Stanford University",
                "UC Berkeley", 
                "Carnegie Mellon University",
                "Harvard University",
                "Princeton University",
                "University of Washington",
                "University of Toronto",
                "ETH Zurich",
                "EPFL",
                "UC San Diego",
                "University of Michigan",
                "Cornell University",
                "University of Illinois",
                "Georgia Institute of Technology",
                # Industry Research Labs
                "NVIDIA Research",
                "Google Research",
                "Microsoft Research",
                "Facebook AI Research",
                "Meta AI",
                "Intel Labs",
                "IBM Research",
                "Apple Machine Learning Research",
                "Amazon Science",
                "OpenAI",
                "DeepMind",
                "Anthropic"
            ],
            "custom_queries": [],
            "relevance_filters": {
                "require_both_topics": True,
                "architecture_keywords": [
                    "architecture", "hardware", "accelerator", "fpga", "gpu",
                    "processor", "chip", "performance", "parallel", "distributed"
                ],
                "ml_keywords": [
                    "machine learning", "deep learning", "neural network", "ai",
                    "model", "inference", "training"
                ]
            }
        }
        
        # 4. General Config
        general_config = {
            "name": "General ArXiv Paper Search",
            "description": "Flexible configuration for general paper searches",
            "search_mode": "general",
            "max_papers": 100,
            "download_dir": "./papers",
            "target_authors": [],
            "target_organizations": [],
            "custom_queries": [
                'cat:cs.LG',  # Machine Learning
                'cat:cs.AI',  # Artificial Intelligence
                'cat:cs.AR',  # Computer Architecture
                'cat:cs.DC',  # Distributed Computing
                'cat:cs.PF'   # Performance
            ],
            "relevance_filters": {
                "require_both_topics": False,
                "architecture_keywords": [],
                "ml_keywords": []
            }
        }
        
        # Save configs if they don't exist
        configs = {
            "comp_arch_ml.yaml": comp_arch_ml_config,
            "authors.yaml": authors_config, 
            "institutions.yaml": institutions_config,
            "general.yaml": general_config
        }
        
        for filename, config in configs.items():
            config_path = self.config_dir / filename
            if not config_path.exists():
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                logger.info(f"Created default config: {config_path}")
    
    def load_config(self, config_name: str) -> Dict:
        """Load configuration from file"""
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'
        
        config_path = self.config_dir / config_name
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded config: {config['name']}")
        return config
    
    def list_configs(self) -> List[str]:
        """List available configuration files"""
        return [f.stem for f in self.config_dir.glob("*.yaml")]


class ArxivScraper:
    def __init__(self, config: Dict):
        """
        Initialize the ArXiv scraper with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.download_dir = Path(config['download_dir'])
        self.max_papers = config['max_papers']
        self.base_url = "http://export.arxiv.org/api/query?"
        self.papers_metadata = []
        self.downloaded_papers: Set[str] = set()
        self.target_authors = config.get('target_authors', [])
        self.target_organizations = config.get('target_organizations', [])
        self.search_mode = config.get('search_mode', 'general')
        
        # Create download directory
        self.download_dir.mkdir(exist_ok=True)
        
        # Build search queries
        self.search_queries = self._build_search_queries()
    
    def _build_search_queries(self) -> List[str]:
        """Build search queries based on configuration"""
        custom_queries = self.config.get('custom_queries', [])
        
        if self.search_mode == "general":
            return custom_queries
        
        queries = custom_queries.copy()
        
        # Author-specific queries
        if self.search_mode in ["authors", "combined"] and self.target_authors:
            for author in self.target_authors:
                author_clean = author.strip()
                queries.extend([
                    f'au:"{author_clean}"',
                    f'au:"{author_clean}" AND cat:cs.AR',
                    f'au:"{author_clean}" AND cat:cs.LG',
                    f'au:"{author_clean}" AND cat:cs.AI',
                    f'au:"{author_clean}" AND "machine learning"',
                    f'au:"{author_clean}" AND "computer architecture"',
                    f'au:"{author_clean}" AND "hardware"',
                    f'au:"{author_clean}" AND "neural network"'
                ])
        
        # Organization-specific queries
        if self.search_mode in ["organizations", "combined"] and self.target_organizations:
            for org in self.target_organizations:
                org_clean = org.strip()
                queries.extend([
                    f'"{org_clean}"',
                    f'"{org_clean}" AND cat:cs.AR',
                    f'"{org_clean}" AND cat:cs.LG',
                    f'"{org_clean}" AND cat:cs.AI',
                    f'"{org_clean}" AND "machine learning"',
                    f'"{org_clean}" AND "computer architecture"'
                ])
        
        return queries
    
    def search_papers(self, query: str, start: int = 0, max_results: int = 100) -> List[Dict]:
        """Search for papers using ArXiv API"""
        params = {
            'search_query': query,
            'start': start,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        url = self.base_url + urlencode(params)
        logger.info(f"Searching with query: {query}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            
            papers = []
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                paper_data = self._parse_paper_entry(entry)
                if paper_data:
                    papers.append(paper_data)
            
            logger.info(f"Found {len(papers)} papers for query: {query}")
            return papers
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching papers: {e}")
            return []
        except ET.ParseError as e:
            logger.error(f"Error parsing XML response: {e}")
            return []
    
    def _parse_paper_entry(self, entry) -> Optional[Dict]:
        """Parse a single paper entry from ArXiv API response"""
        try:
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
            
            id_element = entry.find('{http://www.w3.org/2005/Atom}id')
            arxiv_id = id_element.text.split('/')[-1]
            
            authors = []
            author_affiliations = []
            for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                name = author.find('{http://www.w3.org/2005/Atom}name').text
                authors.append(name)
                
                affiliation_elem = author.find('{http://arxiv.org/schemas/atom}affiliation')
                if affiliation_elem is not None:
                    author_affiliations.append(affiliation_elem.text)
                else:
                    author_affiliations.append("")
            
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            
            categories = []
            for category in entry.findall('{http://www.w3.org/2005/Atom}category'):
                categories.append(category.get('term'))
            
            pdf_link = None
            for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
                if link.get('type') == 'application/pdf':
                    pdf_link = link.get('href')
                    break
            
            return {
                'arxiv_id': arxiv_id,
                'title': title,
                'authors': authors,
                'author_affiliations': author_affiliations,
                'abstract': abstract,
                'published': published,
                'categories': categories,
                'pdf_link': pdf_link
            }
            
        except Exception as e:
            logger.error(f"Error parsing paper entry: {e}")
            return None
    
    def matches_target_criteria(self, paper: Dict) -> bool:
        """Check if paper matches target criteria from config"""
        if self.search_mode == "general":
            return self._check_relevance_filters(paper)
        
        matches_author = False
        matches_org = False
        
        # Check authors
        if self.target_authors:
            paper_authors_text = " ".join(paper['authors']).lower()
            for target_author in self.target_authors:
                author_parts = target_author.lower().split()
                if len(author_parts) >= 2:
                    if all(part in paper_authors_text for part in author_parts):
                        matches_author = True
                        break
                else:
                    if target_author.lower() in paper_authors_text:
                        matches_author = True
                        break
        
        # Check organizations
        if self.target_organizations:
            search_text = (paper['abstract'] + " " + " ".join(paper['author_affiliations'])).lower()
            
            for target_org in self.target_organizations:
                org_variants = [
                    target_org.lower(),
                    target_org.lower().replace("university", "univ"),
                    target_org.lower().replace("univ", "university"),
                    target_org.lower().replace("institute", "inst"),
                    target_org.lower().replace("inst", "institute")
                ]
                
                if any(variant in search_text for variant in org_variants):
                    matches_org = True
                    break
        
        # Apply search mode logic
        if self.search_mode == "authors":
            result = matches_author if self.target_authors else True
        elif self.search_mode == "organizations":
            result = matches_org if self.target_organizations else True
        elif self.search_mode == "combined":
            author_check = matches_author if self.target_authors else True
            org_check = matches_org if self.target_organizations else True
            result = author_check and org_check
        else:
            result = True
        
        # Apply relevance filters
        return result and self._check_relevance_filters(paper)
    
    def _check_relevance_filters(self, paper: Dict) -> bool:
        """Check paper against relevance filters from config"""
        filters = self.config.get('relevance_filters', {})
        
        if not filters.get('require_both_topics', False):
            return True
        
        arch_keywords = filters.get('architecture_keywords', [])
        ml_keywords = filters.get('ml_keywords', [])
        
        if not arch_keywords or not ml_keywords:
            return True
        
        text = (paper['title'] + ' ' + paper['abstract']).lower()
        
        has_arch = any(keyword.lower() in text for keyword in arch_keywords)
        has_ml = any(keyword.lower() in text for keyword in ml_keywords)
        
        return has_arch and has_ml
    
    def download_pdf(self, paper: Dict) -> bool:
        """Download PDF for a paper"""
        if not paper['pdf_link']:
            logger.warning(f"No PDF link for paper: {paper['arxiv_id']}")
            return False
        
        safe_title = re.sub(r'[^\w\s-]', '', paper['title'])
        safe_title = re.sub(r'\s+', '_', safe_title)
        filename = f"{paper['arxiv_id']}_{safe_title[:50]}.pdf"
        filepath = self.download_dir / filename
        
        if filepath.exists():
            logger.info(f"Paper already downloaded: {paper['arxiv_id']}")
            return True
        
        try:
            logger.info(f"Downloading: {paper['title']}")
            response = requests.get(paper['pdf_link'], timeout=60)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded: {filename}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading {paper['arxiv_id']}: {e}")
            return False
    
    def save_metadata(self) -> None:
        """Save paper metadata and config to files"""
        # Save metadata
        metadata_file = self.download_dir / 'papers_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.papers_metadata, f, indent=2, default=str)
        
        # Save config used
        config_file = self.download_dir / 'config_used.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved metadata for {len(self.papers_metadata)} papers")
    
    def run(self) -> None:
        """Main execution function"""
        logger.info(f"Starting ArXiv scraper: {self.config['name']}")
        logger.info(f"Target: {self.max_papers} papers")
        
        all_papers = []
        seen_ids = set()
        
        for query in self.search_queries:
            if len(all_papers) >= self.max_papers:
                break
                
            start = 0
            batch_size = 100
            
            while len(all_papers) < self.max_papers:
                papers = self.search_papers(query, start, batch_size)
                
                if not papers:
                    break
                
                for paper in papers:
                    if len(all_papers) >= self.max_papers:
                        break
                    
                    if paper['arxiv_id'] not in seen_ids:
                        if self.matches_target_criteria(paper):
                            all_papers.append(paper)
                            seen_ids.add(paper['arxiv_id'])
                            
                            criteria_msg = f"Added paper {len(all_papers)}: {paper['title']}"
                            if self.target_authors or self.target_organizations:
                                matching_authors = [a for a in self.target_authors 
                                                  if any(part.lower() in " ".join(paper['authors']).lower() 
                                                       for part in a.split())]
                                if matching_authors:
                                    criteria_msg += f" [Author: {', '.join(matching_authors)}]"
                            
                            logger.info(criteria_msg)
                
                start += batch_size
                time.sleep(1)
        
        logger.info(f"Found {len(all_papers)} unique relevant papers")
        
        # Download papers
        successful_downloads = 0
        for i, paper in enumerate(all_papers, 1):
            logger.info(f"Processing paper {i}/{len(all_papers)}")
            
            if self.download_pdf(paper):
                self.papers_metadata.append(paper)
                successful_downloads += 1
            
            time.sleep(3)  # Rate limiting
        
        self.save_metadata()
        
        logger.info(f"Scraping complete!")
        logger.info(f"Successfully downloaded: {successful_downloads} papers")
        logger.info(f"Papers saved to: {self.download_dir}")
        
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print summary of downloaded papers"""
        print("\n" + "="*60)
        print("ARXIV SCRAPER SUMMARY")
        print("="*60)
        print(f"Configuration: {self.config['name']}")
        print(f"Search mode: {self.search_mode}")
        if self.target_authors:
            print(f"Target authors: {len(self.target_authors)} authors")
        if self.target_organizations:
            print(f"Target organizations: {len(self.target_organizations)} organizations")
        print(f"Total papers downloaded: {len(self.papers_metadata)}")
        print(f"Download directory: {self.download_dir}")
        
        if self.papers_metadata:
            print(f"\nSample papers:")
            for i, paper in enumerate(self.papers_metadata[:5], 1):
                print(f"{i}. {paper['title']}")
                print(f"   Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
                print(f"   ArXiv ID: {paper['arxiv_id']}")
                print()


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="ArXiv Paper Scraper with Configuration Files")
    parser.add_argument('--config', '-c', type=str, 
                       help='Configuration file to use (without .yaml extension)')
    parser.add_argument('--list-configs', action='store_true',
                       help='List available configuration files')
    parser.add_argument('--create-configs', action='store_true',
                       help='Create default configuration files')
    
    args = parser.parse_args()
    
    config_manager = ConfigManager()
    
    if args.list_configs:
        print("Available configurations:")
        for config_name in config_manager.list_configs():
            print(f"  - {config_name}")
        return
    
    if args.create_configs:
        print("Default configuration files created in 'configs/' directory")
        return
    
    # Determine which config to use
    if args.config:
        config_name = args.config
    else:
        # Interactive selection
        configs = config_manager.list_configs()
        print("Available configurations:")
        for i, config in enumerate(configs, 1):
            print(f"{i}. {config}")
        
        while True:
            try:
                choice = int(input(f"\nSelect configuration (1-{len(configs)}): ")) - 1
                if 0 <= choice < len(configs):
                    config_name = configs[choice]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    # Load config and run scraper
    try:
        config = config_manager.load_config(config_name)
        scraper = ArxivScraper(config)
        scraper.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        logger.error(f"Error running scraper: {e}")


if __name__ == "__main__":
    # Install requirements check
    try:
        import yaml
    except ImportError:
        print("Please install required packages:")
        print("pip install requests pyyaml")
        exit(1)
    
    main()
