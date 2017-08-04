sbt package

$SPARK_HOME/bin/spark-submit \
    --master yarn \
    --driver-memory 4g \
    --executor-memory 4G  \
    --num-executors 4 \
    --driver-cores 4  \
    --executor-cores 4 \
    --class TestFFM \
    target/scala-2.11/spark-ffm_2.11-0.0.1.jar \
    train_adasyn100x_shuf.ffm \
    8 \
    10 \
    0.1 \
    0.0002 \
    true \
    true \
    1 \
    valid_adasyn100x.ffm \
    test.ffm \
    true \
    false \
    1.0

    #file:///home/gzsuweiyue/Project/spark-ffm/data/a9a_ffm \
    #/user/gzsuweiyue/Data/netease_ctr/split/20170714/train_adasyn100x.ffm \
    #/user/gzsuweiyue/Data/netease_ctr/split/20170714/valid.ffm \
    #/user/gzsuweiyue/Data/netease_ctr/split/20170714/test.ffm
