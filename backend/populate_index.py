import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("data/faiss_index")
META_PATH = Path("data/faiss_metadata.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

model = SentenceTransformer(EMBEDDING_MODEL)

def get_documents() -> list[str]:
    print("Generating documents list...")

    documents = [
        "This is the full list of risk metrics that are available in the database: \
kurtosis, skewness, sharpe ratio, sortino ratio, volatility, \
var 95 (value at risk 95), var 99 (value at risk 99), cvar 95 (conditional value at risk 95), \
cvar 99 (conditional value at risk 99), max drawdown, mean return, total return, min return, max return, beta"
    ]

    with open("data/companies.json", "r", encoding="utf-8") as f:
        companies = json.load(f)
        companies_dict = {company['ticker']: company for company in companies}
        for company in companies:
            documents.append(f"Company information - ticker: {company['ticker']}, \
exchange: {company['exchange']}, stock name: {company['stock_name']}, \
company name: {company['company_name']}, sector: {company['sector']}, \
industry: {company['industry']}, country: {company['country']}")

    with open("data/risk.json", "r", encoding="utf-8") as f:
        risks = json.load(f)
        for risk in risks:
            risk_metric_name = risk['risk_metric_name']
            risk_metric_name = risk_metric_name.replace("_", " ")
            risk_metric_value = risk['risk_metric_value']
            ticker = risk['ticker']
            if ticker not in companies_dict:
                continue
            company_name = companies_dict[ticker]['company_name']
            documents.append(f"{risk_metric_name} with value {risk_metric_value} \
is a risk that the company {company_name} with ticker {ticker} faces.")

    return documents

def get_vectors(documents: list[str]) -> np.ndarray:
    print("Generating embeddings...")
    embeddings = model.encode(documents, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)

if __name__ == "__main__":
    documents = get_documents()
    vectors = get_vectors(documents)
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    metadata = [{"id": i, "text": text} for i, text in enumerate(documents)]

    faiss.write_index(index, str(INDEX_PATH))

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)