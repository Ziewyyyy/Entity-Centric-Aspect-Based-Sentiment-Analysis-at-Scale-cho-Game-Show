
#!/bin/bash
set -e

BOOTSTRAP_SERVER="kafka:29092"

echo "Waiting for Kafka..."
until kafka-topics \
  --bootstrap-server "$BOOTSTRAP_SERVER" \
  --list >/dev/null 2>&1; do
  sleep 2
done

echo "Creating raw-comment topic..."
kafka-topics \
  --bootstrap-server "$BOOTSTRAP_SERVER" \
  --create \
  --if-not-exists \
  --topic raw-comment \
  --partitions 3 \
  --replication-factor 1 \
  --config retention.ms=604800000 \
  --config cleanup.policy=delete

echo "Creating enriched-comment topic..."
kafka-topics \
  --bootstrap-server "$BOOTSTRAP_SERVER" \
  --create \
  --if-not-exists \
  --topic enriched-comment \
  --partitions 3 \
  --replication-factor 1 \
  --config retention.ms=604800000 \
  --config cleanup.policy=delete

echo "Kafka topics:"
kafka-topics \
  --bootstrap-server "$BOOTSTRAP_SERVER" \
  --list

echo "Kafka topic setup completed!"
