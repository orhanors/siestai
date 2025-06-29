import os
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class IntercomArticle:
    """Data class for Intercom article objects"""
    id: str
    type: str
    workspace_id: str
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
    default_locale: Optional[str]
    translated_content: Optional[Dict]
    tags: Dict
    statistics: Optional[Dict]


@dataclass
class IntercomConversation:
    """Data class for Intercom conversation objects"""
    id: str
    type: str
    created_at: int
    updated_at: int
    source: Dict
    contacts: List[Dict]
    teammates: List[Dict]
    title: Optional[str]
    body: Optional[str]
    author: Dict
    assignee: Optional[Dict]
    open: bool
    read: bool
    waiting_since: Optional[int]
    snoozed_until: Optional[int]
    tags: Dict
    first_contact_reply: Optional[Dict]
    priority: str
    sla_applied: Optional[Dict]
    conversation_rating: Optional[Dict]
    team_assignee_id: Optional[int]
    custom_attributes: Dict


class IntercomDataSource:
    """
    A data source class for interacting with the Intercom API.
    
    This class provides methods to fetch articles and conversations from Intercom
    using their REST API v2.13.
    """
    
    def __init__(self, access_token: Optional[str] = None, base_url: str = "https://api.intercom.io"):
        """
        Initialize the Intercom data source.
        
        Args:
            access_token: Intercom access token. If not provided, will try to get from INTERCOM_ACCESS_TOKEN env var
            base_url: Base URL for Intercom API. Defaults to production API
        """
        self.access_token = access_token or os.getenv("INTERCOM_ACCESS_TOKEN")
        if not self.access_token:
            raise ValueError("Intercom access token is required. Set INTERCOM_ACCESS_TOKEN environment variable or pass it to the constructor.")
        
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Intercom-Version": "2.13",
            "Content-Type": "application/json"
        }
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make a request to the Intercom API.
        
        Args:
            endpoint: API endpoint (e.g., '/articles')
            params: Query parameters
            
        Returns:
            API response as dictionary
            
        Raises:
            requests.RequestException: If the API request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Intercom API request failed: {e}")
    
    def list_articles(
        self, 
        per_page: int = 50, 
        page: Optional[int] = None,
        help_center_id: Optional[int] = None
    ) -> List[IntercomArticle]:
        """
        List all articles from Intercom.
        
        Args:
            per_page: Number of articles per page (max 50)
            page: Page number for pagination
            help_center_id: ID of the help center to filter articles
            
        Returns:
            List of IntercomArticle objects
            
        Reference:
            https://developers.intercom.com/docs/references/rest-api/api.intercom.io/articles/listarticles
        """
        params = {"per_page": min(per_page, 50)}
        
        if page is not None:
            params["page"] = page
        if help_center_id is not None:
            params["help_center_id"] = help_center_id
        
        response = self._make_request("/articles", params=params)
        
        articles = []
        for article_data in response.get("data", []):
            article = IntercomArticle(
                id=article_data.get("id"),
                type=article_data.get("type"),
                workspace_id=article_data.get("workspace_id"),
                title=article_data.get("title"),
                description=article_data.get("description"),
                body=article_data.get("body"),
                author_id=article_data.get("author_id"),
                state=article_data.get("state"),
                created_at=article_data.get("created_at"),
                updated_at=article_data.get("updated_at"),
                url=article_data.get("url"),
                parent_id=article_data.get("parent_id"),
                parent_ids=article_data.get("parent_ids", []),
                parent_type=article_data.get("parent_type"),
                default_locale=article_data.get("default_locale"),
                translated_content=article_data.get("translated_content"),
                tags=article_data.get("tags", {}),
                statistics=article_data.get("statistics")
            )
            articles.append(article)
        
        return articles
    
    def search_articles(
        self,
        phrase: str,
        state: str = "published",
        help_center_id: Optional[int] = None,
        highlight: bool = True
    ) -> Dict[str, Any]:
        """
        Search for articles in Intercom.
        
        Args:
            phrase: The phrase to search for within articles
            state: Article state filter ('published', 'draft', or 'all')
            help_center_id: ID of the help center to search in
            highlight: Whether to return highlighted content
            
        Returns:
            Search results with articles and highlights
            
        Reference:
            https://developers.intercom.com/docs/references/rest-api/api.intercom.io/articles/search
        """
        params = {
            "phrase": phrase,
            "state": state,
            "highlight": highlight
        }
        
        if help_center_id is not None:
            params["help_center_id"] = help_center_id
        
        return self._make_request("/articles/search", params=params)
    
    def list_conversations(
        self,
        per_page: int = 50,
        page: Optional[int] = None,
        open: Optional[bool] = None,
        read: Optional[bool] = None,
        tag_id: Optional[int] = None,
        team_id: Optional[int] = None,
        assignee_id: Optional[int] = None,
        contact_id: Optional[int] = None,
        since: Optional[int] = None
    ) -> List[IntercomConversation]:
        """
        List conversations from Intercom.
        
        Args:
            per_page: Number of conversations per page (max 50)
            page: Page number for pagination
            open: Filter by open/closed status
            read: Filter by read/unread status
            tag_id: Filter by tag ID
            team_id: Filter by team ID
            assignee_id: Filter by assignee ID
            contact_id: Filter by contact ID
            since: Filter conversations created since this timestamp
            
        Returns:
            List of IntercomConversation objects
            
        Reference:
            https://developers.intercom.com/docs/references/rest-api/api.intercom.io/conversations
        """
        params = {"per_page": min(per_page, 50)}
        
        if page is not None:
            params["page"] = page
        if open is not None:
            params["open"] = open
        if read is not None:
            params["read"] = read
        if tag_id is not None:
            params["tag_id"] = tag_id
        if team_id is not None:
            params["team_id"] = team_id
        if assignee_id is not None:
            params["assignee_id"] = assignee_id
        if contact_id is not None:
            params["contact_id"] = contact_id
        if since is not None:
            params["since"] = since
        
        response = self._make_request("/conversations", params=params)
        
        conversations = []
        for conv_data in response.get("conversations", []):
            conversation = IntercomConversation(
                id=conv_data.get("id"),
                type=conv_data.get("type"),
                created_at=conv_data.get("created_at"),
                updated_at=conv_data.get("updated_at"),
                source=conv_data.get("source", {}),
                contacts=conv_data.get("contacts", []),
                teammates=conv_data.get("teammates", []),
                title=conv_data.get("title"),
                body=conv_data.get("body"),
                author=conv_data.get("author", {}),
                assignee=conv_data.get("assignee"),
                open=conv_data.get("open", False),
                read=conv_data.get("read", False),
                waiting_since=conv_data.get("waiting_since"),
                snoozed_until=conv_data.get("snoozed_until"),
                tags=conv_data.get("tags", {}),
                first_contact_reply=conv_data.get("first_contact_reply"),
                priority=conv_data.get("priority", "normal"),
                sla_applied=conv_data.get("sla_applied"),
                conversation_rating=conv_data.get("conversation_rating"),
                team_assignee_id=conv_data.get("team_assignee_id"),
                custom_attributes=conv_data.get("custom_attributes", {})
            )
            conversations.append(conversation)
        
        return conversations
    
    def get_conversation(self, conversation_id: str) -> IntercomConversation:
        """
        Get a specific conversation by ID.
        
        Args:
            conversation_id: The ID of the conversation to retrieve
            
        Returns:
            IntercomConversation object
            
        Reference:
            https://developers.intercom.com/docs/references/rest-api/api.intercom.io/conversations
        """
        response = self._make_request(f"/conversations/{conversation_id}")
        
        conv_data = response.get("conversation", {})
        return IntercomConversation(
            id=conv_data.get("id"),
            type=conv_data.get("type"),
            created_at=conv_data.get("created_at"),
            updated_at=conv_data.get("updated_at"),
            source=conv_data.get("source", {}),
            contacts=conv_data.get("contacts", []),
            teammates=conv_data.get("teammates", []),
            title=conv_data.get("title"),
            body=conv_data.get("body"),
            author=conv_data.get("author", {}),
            assignee=conv_data.get("assignee"),
            open=conv_data.get("open", False),
            read=conv_data.get("read", False),
            waiting_since=conv_data.get("waiting_since"),
            snoozed_until=conv_data.get("snoozed_until"),
            tags=conv_data.get("tags", {}),
            first_contact_reply=conv_data.get("first_contact_reply"),
            priority=conv_data.get("priority", "normal"),
            sla_applied=conv_data.get("sla_applied"),
            conversation_rating=conv_data.get("conversation_rating"),
            team_assignee_id=conv_data.get("team_assignee_id"),
            custom_attributes=conv_data.get("custom_attributes", {})
        )
    
    def search_conversations(
        self,
        query: str,
        per_page: int = 50,
        page: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Search for conversations in Intercom.
        
        Args:
            query: Search query string
            per_page: Number of results per page
            page: Page number for pagination
            
        Returns:
            Search results with conversations
            
        Reference:
            https://developers.intercom.com/docs/references/rest-api/api.intercom.io/conversations
        """
        params = {
            "q": query,
            "per_page": min(per_page, 50)
        }
        
        if page is not None:
            params["page"] = page
        
        return self._make_request("/conversations/search", params=params)