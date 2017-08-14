sbt package

$SPARK_HOME/bin/spark-submit \
    --master yarn \
    --driver-memory 4g \
    --executor-memory 4G  \
    --num-executors 4 \
    --driver-cores 4  \
    --executor-cores 4 \
    --class TestFFMOnce \
    target/scala-2.11/spark-ffm_2.11-0.0.1.jar \
    train.ffm \
    5 \
    1000 \
    0.1 \
    0.0002 \
    false \
    false \
    16 \
    valid.ffm \
    test.ffm \
    true \
    true \
    1.0 \
    1.0 \
    1.0 \
    1.0 \
    1.0 \
    adag \
    /user/gzsuweiyue/Model/ftrl

