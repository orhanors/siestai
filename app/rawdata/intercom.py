import os
from dotenv import load_dotenv
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

load_dotenv()


@dataclass
class IntercomArticle:
    """Data class representing an Intercom article"""
    id: str
    title: str
    description: Optional[str]
    body: Optional[str]
    author_id: int
    state: str
    created_at: int
    updated_at: int
    url: Optional[str]
    parent_id: Optional[int]
    parent_ids: List[int]
    parent_type: Optional[str]
    default_locale: str
    workspace_id: str
    type: str = "article"
    statistics: Optional[Dict] = None
    translated_content: Optional[Dict] = None
    tags: Optional[Dict] = None


class IntercomDataProvider:
    """
    A data provider class for interacting with the Intercom API.
    
    This class handles authentication and provides methods to interact with
    various Intercom API endpoints including articles, contacts, conversations,
    and more.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.intercom.io"):
        """
        Initialize the IntercomDataProvider.
        
        Args:
            api_key: Intercom API key. If not provided, will try to get from INTERCOM_KEY env var
            base_url: Base URL for Intercom API. Defaults to production API
        """
        self.api_key = api_key or os.getenv('INTERCOM_KEY')
        if not self.api_key:
            raise ValueError("Intercom API key is required. Set INTERCOM_KEY environment variable or pass api_key parameter.")
        
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Intercom-Version': '2.13',
            'Content-Type': 'application/json'
        })
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                     data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make a request to the Intercom API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            params: Query parameters
            data: Request body data
            
        Returns:
            API response as dictionary
            
        Raises:
            requests.RequestException: If the request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Intercom API request failed: {e}")
    
    def get_articles(self, per_page: int = 50, page: Optional[int] = None, 
                    state: Optional[str] = None) -> List[IntercomArticle]:
        """
        Get all articles from Intercom.
        
        Args:
            per_page: Number of articles per page (max 50)
            page: Page number for pagination
            state: Filter by state ('published', 'draft', or None for all)
            
        Returns:
            List of IntercomArticle objects
        """
        params = {'per_page': min(per_page, 50)}
        
        if page:
            params['page'] = page
        if state:
            params['state'] = state
        
        response = self._make_request('GET', '/articles', params=params)
        
        articles = []
        print("Response pagessss: ", response["pages"])
        for article_data in response.get('data', []):
            article = IntercomArticle(
                id=article_data['id'],
                title=article_data['title'],
                description=article_data.get('description'),
                body=article_data.get('body'),
                author_id=article_data['author_id'],
                state=article_data['state'],
                created_at=article_data['created_at'],
                updated_at=article_data['updated_at'],
                url=article_data.get('url'),
                parent_id=article_data.get('parent_id'),
                parent_ids=article_data.get('parent_ids', []),
                parent_type=article_data.get('parent_type'),
                default_locale=article_data.get('default_locale', 'en'),
                workspace_id=article_data['workspace_id'],
                type=article_data['type'],
                statistics=article_data.get('statistics'),
                translated_content=article_data.get('translated_content'),
                tags=article_data.get('tags')
            )
            articles.append(article)
        
        next_page = None
        if 'pages' in response and response['pages']:
            next_page = response['pages'].get('next')
            
        return articles, next_page
    
    def get_article(self, article_id: str) -> IntercomArticle:
        """
        Get a specific article by ID.
        
        Args:
            article_id: The ID of the article to retrieve
            
        Returns:
            IntercomArticle object
        """
        response = self._make_request('GET', f'/articles/{article_id}')
        
        article_data = response
        return IntercomArticle(
            id=article_data['id'],
            title=article_data['title'],
            description=article_data.get('description'),
            body=article_data.get('body'),
            author_id=article_data['author_id'],
            state=article_data['state'],
            created_at=article_data['created_at'],
            updated_at=article_data['updated_at'],
            url=article_data.get('url'),
            parent_id=article_data.get('parent_id'),
            parent_ids=article_data.get('parent_ids', []),
            parent_type=article_data.get('parent_type'),
            default_locale=article_data.get('default_locale', 'en'),
            workspace_id=article_data['workspace_id'],
            type=article_data['type'],
            statistics=article_data.get('statistics'),
            translated_content=article_data.get('translated_content'),
            tags=article_data.get('tags')
        )
    
    def search_articles(self, phrase: str, state: Optional[str] = None, 
                       help_center_id: Optional[int] = None, 
                       highlight: bool = True) -> Dict[str, Any]:
        """
        Search for articles using the Intercom search endpoint.
        
        Args:
            phrase: The search phrase
            state: Filter by state ('published', 'draft', 'all')
            help_center_id: ID of the help center to search in
            highlight: Whether to return highlighted content
            
        Returns:
            Search results dictionary
        """
        params = {
            'phrase': phrase,
            'highlight': highlight
        }
        
        if state:
            params['state'] = state
        if help_center_id:
            params['help_center_id'] = help_center_id
        
        return self._make_request('GET', '/articles/search', params=params)
    
    def get_contacts(self, per_page: int = 50, page: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all contacts from Intercom.
        
        Args:
            per_page: Number of contacts per page (max 50)
            page: Page number for pagination
            
        Returns:
            List of contact dictionaries
        """
        params = {'per_page': min(per_page, 50)}
        
        if page:
            params['page'] = page
        
        response = self._make_request('GET', '/contacts', params=params)
        return response.get('data', [])
    
    def get_contact(self, contact_id: str) -> Dict[str, Any]:
        """
        Get a specific contact by ID.
        
        Args:
            contact_id: The ID of the contact to retrieve
            
        Returns:
            Contact dictionary
        """
        return self._make_request('GET', f'/contacts/{contact_id}')
    
    def get_conversations(self, per_page: int = 50, page: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all conversations from Intercom.
        
        Args:
            per_page: Number of conversations per page (max 50)
            page: Page number for pagination
            
        Returns:
            List of conversation dictionaries
        """
        params = {'per_page': min(per_page, 50)}
        
        if page:
            params['page'] = page
        
        response = self._make_request('GET', '/conversations', params=params)
        return response.get('data', [])
    
    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get a specific conversation by ID.
        
        Args:
            conversation_id: The ID of the conversation to retrieve
            
        Returns:
            Conversation dictionary
        """
        return self._make_request('GET', f'/conversations/{conversation_id}')
    
    def get_companies(self, per_page: int = 50, page: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all companies from Intercom.
        
        Args:
            per_page: Number of companies per page (max 50)
            page: Page number for pagination
            
        Returns:
            List of company dictionaries
        """
        params = {'per_page': min(per_page, 50)}
        
        if page:
            params['page'] = page
        
        response = self._make_request('GET', '/companies', params=params)
        return response.get('data', [])
    
    def get_company(self, company_id: str) -> Dict[str, Any]:
        """
        Get a specific company by ID.
        
        Args:
            company_id: The ID of the company to retrieve
            
        Returns:
            Company dictionary
        """
        return self._make_request('GET', f'/companies/{company_id}')
    
    def get_help_center_collections(self) -> List[Dict[str, Any]]:
        """
        Get all help center collections.
        
        Returns:
            List of collection dictionaries
        """
        response = self._make_request('GET', '/help_center/collections')
        return response.get('data', [])
    
    def get_help_centers(self) -> List[Dict[str, Any]]:
        """
        Get all help centers.
        
        Returns:
            List of help center dictionaries
        """
        response = self._make_request('GET', '/help_center/help_centers')
        return response.get('data', [])
    
    def get_tags(self) -> List[Dict[str, Any]]:
        """
        Get all tags.
        
        Returns:
            List of tag dictionaries
        """
        response = self._make_request('GET', '/tags')
        return response.get('data', [])
    
    def get_teams(self) -> List[Dict[str, Any]]:
        """
        Get all teams.
        
        Returns:
            List of team dictionaries
        """
        response = self._make_request('GET', '/teams')
        return response.get('data', [])
    
    def get_admins(self) -> List[Dict[str, Any]]:
        """
        Get all admins.
        
        Returns:
            List of admin dictionaries
        """
        response = self._make_request('GET', '/admins')
        return response.get('data', [])
    
    def get_me(self) -> Dict[str, Any]:
        """
        Get current admin information.
        
        Returns:
            Admin information dictionary
        """
        return self._make_request('GET', '/me')
    
    def create_article(self, title: str, body: str, author_id: int, 
                      description: Optional[str] = None, 
                      state: str = "draft") -> Dict[str, Any]:
        """
        Create a new article.
        
        Args:
            title: Article title
            body: Article body in HTML
            author_id: ID of the author
            description: Article description
            state: Article state ('draft' or 'published')
            
        Returns:
            Created article dictionary
        """
        data = {
            'title': title,
            'body': body,
            'author_id': author_id,
            'state': state
        }
        
        if description:
            data['description'] = description
        
        return self._make_request('POST', '/articles', data=data)
    
    def update_article(self, article_id: str, **kwargs) -> Dict[str, Any]:
        """
        Update an existing article.
        
        Args:
            article_id: ID of the article to update
            **kwargs: Fields to update (title, body, description, state, etc.)
            
        Returns:
            Updated article dictionary
        """
        return self._make_request('PUT', f'/articles/{article_id}', data=kwargs)
    
    def delete_article(self, article_id: str) -> Dict[str, Any]:
        """
        Delete an article.
        
        Args:
            article_id: ID of the article to delete
            
        Returns:
            Deletion response dictionary
        """
        return self._make_request('DELETE', f'/articles/{article_id}')
    
    def get_article_statistics(self, article_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific article.
        
        Args:
            article_id: ID of the article
            
        Returns:
            Article statistics dictionary
        """
        article = self.get_article(article_id)
        return article.statistics or {}
    
    def format_article_for_display(self, article: IntercomArticle) -> str:
        """
        Format an article for display purposes.
        
        Args:
            article: IntercomArticle object
            
        Returns:
            Formatted string representation of the article
        """
        created_date = datetime.fromtimestamp(article.created_at).strftime('%Y-%m-%d %H:%M:%S')
        updated_date = datetime.fromtimestamp(article.updated_at).strftime('%Y-%m-%d %H:%M:%S')
        
        return f"""
Article: {article.title}
ID: {article.id}
State: {article.state}
Author ID: {article.author_id}
Created: {created_date}
Updated: {updated_date}
URL: {article.url or 'N/A'}
Description: {article.description or 'N/A'}
Parent: {article.parent_type or 'None'} (ID: {article.parent_id or 'N/A'})
        """.strip()


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the IntercomDataProvider
    try:
        # Initialize the provider (will use INTERCOM_KEY from environment)
        provider = IntercomDataProvider()
        
        # Get all articles
        articles = provider.get_articles(per_page=10)
        print(f"Found {len(articles)} articles")
        
        # Get a specific article
        if articles:
            first_article = provider.get_article(articles[0].id)
            print(provider.format_article_for_display(first_article))
        
        # Search for articles
        search_results = provider.search_articles("getting started")
        print(f"Search found {search_results.get('total_count', 0)} results")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to set the INTERCOM_KEY environment variable")
