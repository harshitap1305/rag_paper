import requests
import re
import streamlit as st
from config import CROSSREF_API

class CitationSearcher:
    """Searches for citations and related papers"""
    
    @staticmethod
    def search_related_papers(query, limit=5):
        """Search for related papers using Crossref API"""
        try:
            clean_query = re.sub(r'[^\w\s]', ' ', query)
            
            params = {
                'query': clean_query,
                'rows': limit,
                'sort': 'relevance',
                'filter': 'type:journal-article'
            }
            
            response = requests.get(CROSSREF_API, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                papers = []
                
                for item in data.get('message', {}).get('items', []):
                    paper = {
                        'title': item.get('title', ['Unknown'])[0] if item.get('title') else 'Unknown',
                        'authors': [f"{author.get('given', '')} {author.get('family', '')}" 
                                  for author in item.get('author', [])[:3]],
                        'journal': item.get('container-title', ['Unknown'])[0] if item.get('container-title') else 'Unknown',
                        'year': item.get('published-print', {}).get('date-parts', [[2023]])[0][0] if item.get('published-print') else 'Unknown',
                        'doi': item.get('DOI', 'No DOI'),
                        'url': f"https://doi.org/{item.get('DOI')}" if item.get('DOI') else None
                    }
                    papers.append(paper)
                
                return papers
            
        except Exception as e:
            st.warning(f"Citation search failed: {str(e)}")
            return []
        
        return []