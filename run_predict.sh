sbt package

$SPARK_HOME/bin/spark-submit \
    --master yarn \
    --driver-memory 4g \
    --executor-memory 4G  \
    --num-executors 4 \
    --driver-cores 4  \
    --executor-cores 4 \
    --class PredictFFM \
    target/scala-2.11/spark-ffm_2.11-0.0.1.jar \
