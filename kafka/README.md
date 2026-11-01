# Kafka Infrastructure – ABSA Game Show

## 1. Giới thiệu

Hệ thống Kafka phục vụ pipeline xử lý bình luận YouTube và phân tích cảm xúc theo thực thể (Entity-Centric ABSA).

**Công nghệ:** Apache Kafka 7.6.0 (Confluent Platform), Zookeeper, Docker Compose.

## 2. Yêu cầu

- Docker Desktop
- Docker Compose
- Docker Engine đang chạy

## 3. Khởi chạy

Tại thư mục `kafka/`:

```powershell
docker compose up -d
```

Kiểm tra container:

```powershell
docker compose ps
```

## 4. Tạo topic

Chạy trong PowerShell:

```powershell
Get-Content .\create-topics.sh -Raw | docker exec -i absa-kafka bash -s
```

Hai topic:

| Topic | Partitions | Replication | Retention |
|---|---|---|---|
| raw-comment | 3 | 1 | 7 ngày |
| enriched-comment | 3 | 1 | 7 ngày |

## 5. Kiểm tra topic

```powershell
docker exec absa-kafka kafka-topics --bootstrap-server kafka:29092 --list
```

## 6. Kết nối

| Môi trường | Bootstrap servers |
|---|---|
| Windows host | localhost:9092 |
| Docker network | kafka:29092 |

## 7. Consumer groups

Quy ước:

- `spark-cleaning-group`: dành cho Spark đọc raw-comment
- `absa-result-group`: dành cho dịch vụ lưu kết quả enriched-comment
- `test-raw-consumer`: dành cho kiểm thử CLI

Lưu ý: Spark Structured Streaming có cơ chế quản lý offset và checkpoint riêng; không nên dựa vào tên group để thay thế checkpoint.

## 8. Dừng hệ thống

```powershell
docker compose down
```

Khởi chạy lại:

```powershell
docker compose up -d
```

Dữ liệu Kafka được lưu trong Docker volumes.

**Cảnh báo:** `docker compose down -v` xóa cả volumes và dữ liệu đã lưu.

## 9. Trạng thái

Kafka Infrastructure là thành phần đầu vào cho Spark Structured Streaming và pipeline ABSA. Hệ thống sử dụng một broker, phù hợp môi trường phát triển và demo, không phải cấu hình production.