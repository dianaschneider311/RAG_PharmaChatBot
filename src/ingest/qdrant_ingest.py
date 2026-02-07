from src.config.settings import settings
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from src.config.settings import settings
from src.ingest.csv_ingest import load_clean_csv
from src.retrieval.embedder import embed_texts  # your embed function

COLLECTION = settings.QDRANT__COLLECTION

def main():
    df = load_clean_csv(settings.CSV_INPUT_PATH)
    client = QdrantClient(
        url=settings.QDRANT__URL,
        api_key=settings.QDRANT__API_KEY,
    )

    texts = df["document_text"].tolist()
    vectors = embed_texts(texts)

    points = []
    for i, row in df.iterrows():
        payload = row.drop(["document_text"]).to_dict()

        points.append(
            PointStruct(
                id=row["row_id"],
                vector=vectors[i],
                payload=payload,
            )
        )

    client.upsert(
        collection_name=COLLECTION,
        points=points,
    )

    print(f"Upserted {len(points)} points into Qdrant")

if __name__ == "__main__":
    main()


 