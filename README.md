# Entity-Centric-Aspect-Based-Sentiment-Analysis-at-Scale-cho-Game-Show
Hệ thống phân tích cảm xúc theo thực thể và khía cạnh trên luồng bình luận Game Show
## Thành viên: 
1.Trịnh Quang Anh - 24520132
2.Nguyễn Quốc Triệu - 24521855
## Kiến trúc:
             DATA COLLECTION
                    │
          YouTube Game Show
                    │
              Comment Crawler
                    │
                    ▼
             ┌─────────────┐
             │    Kafka    │
             │ raw-comment │
             └──────┬──────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Spark Structured      │
        │ Streaming             │
        │                       │
        │ Cleaning              │
        │ Deduplication         │
        │ Normalization         │
        │ Windowing             │
        └───────────┬───────────┘
                    │
                    ▼
              ABSA AI SERVICE
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
      Entity      Aspect     Opinion Span
    Extraction   Detection    Extraction
        │           │           │
        └───────────┼───────────┘
                    ▼
               Sentiment
              Classification
                    │
                    ▼
              Enriched Kafka
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
    Elasticsearch          PostgreSQL
          │                    │
          └─────────┬──────────┘
                    ▼
                FastAPI
                    │
                    ▼
              Web Dashboard
## Technology Stack:
**Big Data:**
-Apache Kafka – streaming message broker
-Apache Spark Structured Streaming – real-time stream processing
**Artificial Intelligence / NLP:**
-Python
-PyTorch
-Hugging Face Transformers
-PhoBERT / XLM-R
-Vietnamese NLP preprocessing
**Backend:**
-FastAPI
-PostgreSQL
-Elasticsearch
**Frontend:**
-React
-Recharts / Chart.js
**Deployment:**
-Docker
