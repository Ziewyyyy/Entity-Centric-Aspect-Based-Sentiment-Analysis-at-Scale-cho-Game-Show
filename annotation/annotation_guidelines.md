# Annotation Guidelines – Game Show ABSA

## 1. Mục tiêu

Gán nhãn bình luận theo cấu trúc:

**Entity → Aspect → Opinion Span → Sentiment**

## 2. Entity

- ARTIST: Nghệ sĩ tham gia chương trình
- TEAM: Đội thi
- PERFORMANCE: Tiết mục biểu diễn
- SHOW: Chương trình

## 3. Aspect

- VOCAL: Giọng hát
- DANCE: Vũ đạo
- PERFORMANCE: Kỹ năng trình diễn
- VISUAL: Ngoại hình, trang phục
- PERSONALITY: Tính cách, thái độ
- INTERACTION: Tương tác giữa nghệ sĩ
- CONTENT: Nội dung tiết mục
- PRODUCTION: Âm thanh, ánh sáng, sân khấu
- EDITING: Cách dựng và biên tập video

## 4. Sentiment

- POS: Tích cực
- NEG: Tiêu cực
- NEU: Trung lập

## 5. Quy tắc

1. Một bình luận có thể chứa nhiều annotations.
2. Mỗi annotation phải liên kết một Entity với một Aspect.
3. Opinion Span phải là đoạn văn bản xuất hiện thực tế trong bình luận.
4. Không suy đoán sentiment chỉ dựa vào tên nghệ sĩ.
5. Nếu không xác định được thực thể hoặc khía cạnh, đánh dấu để rà soát thay vì tự gán.
6. Giữ nguyên bình luận không có ý kiến rõ ràng với danh sách annotations rỗng.
7. Các trường hợp mỉa mai, phủ định hoặc nhiều ý kiến trái chiều phải được ghi chú và đối chiếu giữa hai người.

## 6. Ví dụ

Bình luận: "RHYDER hát rất hay nhưng nhảy chưa đều."

Annotation 1:
- Entity: RHYDER
- Aspect: VOCAL
- Opinion Span: "hát rất hay"
- Sentiment: POS

Annotation 2:
- Entity: RHYDER
- Aspect: DANCE
- Opinion Span: "nhảy chưa đều"
- Sentiment: NEG
