import unittest
from unittest.mock import Mock, patch

from crawler.kafka_producer import enqueue_message, normalize_comment


class ProducerTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "comment_id": " comment-1 ",
            "video_id": "video-1",
            "text": "  Giọng hát hay ❤️  ",
            "published_at": "2026-11-20T15:00:00+07:00",
        }

    def test_legacy_record_preserves_text_and_normalizes_timestamp(self):
        message = normalize_comment(self.record)
        self.assertEqual(message["comment_id"], "comment-1")
        self.assertEqual(message["text"], self.record["text"])
        self.assertEqual(message["published_at"], "2026-11-20T08:00:00Z")
        self.assertIsNone(message["author"])
        self.assertEqual(len(message), 5)
        self.assertEqual(self.record["comment_id"], " comment-1 ")

    def test_invalid_records_are_rejected(self):
        invalid_records = [
            [],
            {**self.record, "comment_id": " "},
            {**self.record, "video_id": 123},
            {**self.record, "text": ""},
            {**self.record, "text": "bad\ud800"},
            {**self.record, "published_at": "invalid-date"},
            {**self.record, "published_at": "2026-11-20T08:00:00"},
            {**self.record, "author": {}},
        ]
        for record in invalid_records:
            with self.subTest(record=record), self.assertRaises(ValueError):
                normalize_comment(record)

    def test_queue_backpressure_retries_without_changing_payload(self):
        producer = Mock()
        producer.produce.side_effect = [BufferError(), None]
        callback = Mock()
        message = normalize_comment(self.record)
        enqueue_message(producer, "raw-comment", message, callback, 10)
        self.assertEqual(producer.produce.call_count, 2)
        self.assertEqual(producer.produce.call_args_list[0], producer.produce.call_args_list[1])
        kwargs = producer.produce.call_args.kwargs
        self.assertEqual(kwargs["key"], b"comment-1")
        self.assertIn("❤️", kwargs["value"].decode("utf-8"))
        producer.poll.assert_called_once_with(0.1)
        callback.assert_not_called()

    def test_full_queue_has_bounded_wait(self):
        producer = Mock()
        producer.produce.side_effect = BufferError()
        with patch("crawler.kafka_producer.time.monotonic", side_effect=[0, 11]):
            with self.assertRaises(TimeoutError):
                enqueue_message(producer, "raw-comment", normalize_comment(self.record), Mock(), 10)


if __name__ == "__main__":
    unittest.main()
