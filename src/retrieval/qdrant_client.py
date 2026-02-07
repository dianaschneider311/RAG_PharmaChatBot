"""Qdrant vector database client initialization and connection management."""

from qdrant_client import QdrantClient
from src.config.settings import settings


def get_qdrant_client() -> QdrantClient:
    """
    Initialize and return a Qdrant client.

    Returns:
        QdrantClient: Connected Qdrant client instance.

    Raises:
        ValueError: If Qdrant configuration is missing or invalid.
    """
    if not settings.QDRANT__URL or not settings.QDRANT__API_KEY:
        raise ValueError(
            "Qdrant configuration missing. Set QDRANT__URL and QDRANT__API_KEY in .env"
        )

    client = QdrantClient(
        url=settings.QDRANT__URL,
        api_key=settings.QDRANT__API_KEY,
        timeout=30.0,
    )
    return client


def test_connection() -> None:
    """Test connectivity to Qdrant database."""
    try:
        client = get_qdrant_client()
        # Try to list collections to verify connection
        collections = client.get_collections()
        print(f"✅ Qdrant connection OK. Collections: {len(collections.collections)}")
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        raise


if __name__ == "__main__":
    test_connection()
