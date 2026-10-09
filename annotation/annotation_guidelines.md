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

-GENERAL: Đánh giá, khen hoặc chê chung về nghệ sĩ, nhóm, bài hát hoặc tiết mục mà không đề cập khía cạnh cụ thể.
-VOCAL: Giọng hát, giọng rap, kỹ thuật thanh nhạc, cách xử lý giọng.
-DANCE: Vũ đạo, động tác nhảy, kỹ thuật nhảy, độ đồng đều.
-PERFORMANCE: Kỹ năng trình diễn, biểu cảm sân khấu, thần thái biểu diễn, khả năng làm chủ sân khấu.
-VISUAL: Ngoại hình, nhan sắc, trang phục, kiểu tóc, phong cách tạo hình.
-PERSONALITY: Tính cách, thái độ, cách ứng xử của nghệ sĩ.
-INTERACTION: Sự tương tác, phối hợp và ăn ý giữa các nghệ sĩ.
-CONTENT: Nội dung và chất lượng âm nhạc của tiết mục, bao gồm lời bài hát, giai điệu, hòa âm, thông điệp và ý tưởng nghệ thuật.
-PRODUCTION: Chất lượng sản xuất, âm thanh, ánh sáng, hiệu ứng, đạo cụ và thiết kế sân khấu.
-EDITING: Cách quay dựng, góc máy, chuyển cảnh và biên tập video.

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

## 7.Cách sử dụng
Bước 1: Khởi động Label Studio
http://localhost:8080
Vào setting->labeling interface->code gắn:
<View>
  <Header value="Game Show - Entity Centric ABSA"/>

  <Header value="Comment"/>
  <Text name="comment" value="$text"/>

  <Header value="1. Entity and Opinion Span"/>

  <Labels name="spans" toName="comment">
    <Label value="ARTIST" background="#60A5FA"/>
    <Label value="TEAM" background="#A78BFA"/>
    <Label value="SHOW" background="#FBBF24"/>
    <Label value="PERFORMANCE" background="#34D399"/>
    <Label value="OPINION" background="#FB7185"/>
  </Labels>

  <Relations>
    <Relation value="HAS_OPINION"/>
  </Relations>

  <Header value="2. Aspect of Opinion Span"/>

  <Choices
    name="aspect"
    toName="comment"
    perRegion="true"
    choice="single">

    <Choice value="VOCAL"/>
    <Choice value="DANCE"/>
    <Choice value="PERFORMANCE"/>
    <Choice value="VISUAL"/>
    <Choice value="PERSONALITY"/>
    <Choice value="INTERACTION"/>
    <Choice value="CONTENT"/>
    <Choice value="PRODUCTION"/>
    <Choice value="EDITING"/>
    <Choice value="POPULARITY"/>
    <Choice value="AWARD_RESULT"/>
  </Choices>

  <Header value="3. Sentiment of Opinion Span"/>

  <Choices
    name="sentiment"
    toName="comment"
    perRegion="true"
    choice="single">

    <Choice value="POS"/>
    <Choice value="NEG"/>
    <Choice value="NEU"/>
  </Choices>
</View>
Bước 2: Tạo project mới, import file data/labeled/label_studio_tasks.json
Bước 3: Sau khi gán entity, opinion, tạo Relation giữa Entity và Opinion Span và gán quan hệ HAS_OPINION. Hướng quan hệ phải từ Entity đến Opinion Span.
Bước 4: Submit 
Bước 5: sau khi xong 200 comment export ra file json, để vào folder data/labeled
Nếu comment ko rõ ràng thì chỉ submit ko chọn gì cả
