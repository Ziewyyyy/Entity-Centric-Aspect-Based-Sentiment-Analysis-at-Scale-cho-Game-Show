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
Get-Content .\create-topics.sh -Raw | docker exec -i absa-kafka bash -c "tr -d '\r' | bash"
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

## Kafka Producer cho bình luận YouTube

Chạy các lệnh dưới đây từ thư mục gốc dự án. Cài client vào môi trường Python của dự án.

```powershell
.\.venv\Scripts\python.exe -m pip install "confluent-kafka>=2.6,<3"
```

Crawler lấy API key từ `YOUTUBE_API_KEY` trong `.env`, ghi JSONL vào
`data/raw/performance/`. Danh sách `NEW_VIDEO_IDS` trong crawler quyết định các
video được crawl trong lần chạy.

```powershell
.\.venv\Scripts\python.exe .\crawler\youtube_crawler.py --limit 20
.\crawler\run_producer.ps1
```

Gửi một file cụ thể và chọn file log riêng.

```powershell
.\crawler\run_producer.ps1 -InputPath .\data\raw\performance\qKTqFeRb43w.jsonl -LogFile .\logs\producer.log
```

Đầu vào có thể là một file JSONL hoặc thư mục chứa các file `*.jsonl`.
Producer gửi vào `raw-comment`, dùng `comment_id` mã hóa UTF-8 làm key và
JSON UTF-8 làm value. Schema có năm trường sau.

| Trường | Kiểu và quy tắc |
|---|---|
| `comment_id` | Chuỗi khác rỗng, loại khoảng trắng hai đầu |
| `video_id` | Chuỗi khác rỗng, loại khoảng trắng hai đầu |
| `text` | Chuỗi có nội dung, giữ văn bản gốc và emoji |
| `published_at` | Thời gian ISO 8601 có timezone, chuyển về UTC |
| `author` | Chuỗi hoặc `null` với dữ liệu cũ thiếu tác giả |

Các bản ghi lỗi JSON, UTF-8 hoặc thiếu trường bắt buộc được ghi log kèm file và
số dòng. Producer tiếp tục xử lý những dòng còn lại. `SEND_OK` chỉ được ghi sau
khi callback xác nhận giao message thành công. Dòng `SUMMARY` thống kê `read`,
`success`, `invalid`, `failed`, `pending`, `file_errors` và trạng thái ngắt.
`pending` biểu thị số message chưa có kết quả khi kết thúc thời gian chờ.
Mã thoát là 0 khi có dữ liệu và mọi bản ghi đều được gửi thành công, là 1 khi
có lỗi, dữ liệu rỗng hoặc tiến trình bị ngắt. Lỗi tham số CLI trả mã 2.

Cấu hình gửi gồm `acks=all`, `enable.idempotence=true`, `retries=10`,
`retry.backoff.ms=500` và `delivery.timeout.ms=60000`. Thời gian chờ hàng đợi
được giới hạn. Khi kết thúc, Producer gọi `flush()` để chờ callback.
Idempotence xử lý trùng do retry ở cấp giao thức. Mỗi lần chạy script vẫn đọc
lại các file đầu vào, vì vậy gửi lại cùng file có thể tạo bản ghi trùng trên topic.
Các bước xử lý và lưu trữ phía sau cần chống trùng theo khóa đã thống nhất.

Host dùng `-BootstrapServers localhost:9092`. Python chạy trong cùng Docker
network với Kafka dùng `--bootstrap-servers kafka:29092`. Có thể chọn topic
bằng `-Topic` và giới hạn thời gian giao bằng `-DeliveryTimeoutMs`.

### Kiểm thử phần B

File `tests/fixtures/youtube_comments.jsonl` chứa ba bình luận tổng hợp, gồm
tiếng Việt, emoji, timestamp có múi giờ và bản ghi cũ thiếu tác giả.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\crawler\run_producer.ps1 -InputPath .\tests\fixtures\youtube_comments.jsonl -LogFile .\logs\producer_success.log
```

Đọc lại bằng CLI trong terminal khác.

```powershell
docker exec -it absa-kafka kafka-console-consumer --bootstrap-server kafka:29092 --topic raw-comment --from-beginning --property print.key=true --property print.partition=true --property print.offset=true
```

Đối chiếu `comment_id`, nội dung tiếng Việt và tổng `success` với đợt gửi.
`--from-beginning` có thể hiển thị cả message của các lần chạy trước.
Khi kiểm thử broker không truy cập được, dùng một cổng chưa có broker và thời
gian chờ ngắn. Kết quả cần là `failed=3`, `success=0`, `pending=0`, mã thoát 1.

```powershell
.\crawler\run_producer.ps1 -InputPath .\tests\fixtures\youtube_comments.jsonl -BootstrapServers localhost:19092 -DeliveryTimeoutMs 3000 -LogFile .\logs\producer_unreachable.log
```

## 9. Trạng thái

Kafka Infrastructure là thành phần đầu vào cho Spark Structured Streaming và pipeline ABSA. Hệ thống sử dụng một broker, phù hợp môi trường phát triển và demo, không phải cấu hình production.
