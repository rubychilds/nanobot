"""Tests for WebSearchTool per-request env handling.

Verifies that WebSearchTool picks up BRAVE_API_KEY from the per-request
env dict (passed by Orbis via extra_env) when self.api_key is empty.
"""

from unittest.mock import AsyncMock, patch

import pytest

from nanobot.agent.tools.web import WebSearchTool


@pytest.mark.asyncio
async def test_uses_env_brave_key_when_self_key_empty():
    """BRAVE_API_KEY from env dict should be used when init key is empty."""
    tool = WebSearchTool(api_key="")

    with patch.object(tool, "_search_brave", new_callable=AsyncMock, return_value="results") as mock:
        await tool.execute(query="test", env={"BRAVE_API_KEY": "from-env"})
        mock.assert_called_once_with("test", 5, api_key="from-env")


@pytest.mark.asyncio
async def test_self_key_takes_precedence_over_env():
    """Init api_key should take precedence over env dict."""
    tool = WebSearchTool(api_key="from-init")

    with patch.object(tool, "_search_brave", new_callable=AsyncMock, return_value="results") as mock:
        await tool.execute(query="test", env={"BRAVE_API_KEY": "from-env"})
        mock.assert_called_once_with("test", 5, api_key="from-init")


@pytest.mark.asyncio
async def test_error_when_no_key_anywhere():
    """Should return error message when no BRAVE_API_KEY is available."""
    tool = WebSearchTool(api_key="")
    result = await tool.execute(query="test")
    assert "BRAVE_API_KEY not configured" in result


@pytest.mark.asyncio
async def test_env_key_reaches_search_brave():
    """The api_key parameter should be used in the actual HTTP header."""
    tool = WebSearchTool(api_key="")

    with patch("httpx.AsyncClient") as MockClient:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = AsyncMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client

        await tool.execute(query="test query", env={"BRAVE_API_KEY": "test-key-123"})

        mock_client.get.assert_called_once()
        call_kwargs = mock_client.get.call_args
        assert call_kwargs.kwargs["headers"]["X-Subscription-Token"] == "test-key-123"
