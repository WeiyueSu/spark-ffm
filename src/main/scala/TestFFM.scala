import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics



object TestFFM extends App {
  def load_ffm(sc: SparkContext, data_path: String): RDD[(Double, Array[(Int, Int, Double)])] = {
    val data = sc.textFile(data_path).map(_.split("\\s")).map(x => {
        val y = if(x(0).toInt > 0 ) 1.0 else -1.0
        val nodeArray: Array[(Int, Int, Double)] = x.drop(1).map(_.split(":")).map(x => {
            (x(0).toInt, x(1).toInt, x(2).toDouble)
          })
        (y, nodeArray)
      })
    return data
  }

  def train_one_day(sc: SparkContext, args: Array[String], date: Int, initWeights: Vector, m: Int=22, n: Int=7671): FFMModel = {

    val pre_path: String = "/user/gzsuweiyue/Data/netease_ctr/split/201707" + date.toString + "/"
    val partition_num = args(7).toInt
    val train_data = load_ffm(sc, pre_path + args(0)).repartition(partition_num)
    val valid_data = load_ffm(sc, pre_path + args(8)).repartition(partition_num)
    val test_data = load_ffm(sc, pre_path + args(9)).repartition(partition_num)

    //val data = train_data.union(valid_data)
    //sometimes the max feature/field number would be different in train_data/valid_data dataset,
    // so use the whole dataset to get the max feature/field number

    val ffm: FFMModel = FFMWithAdag.train(train_data, m, n, dim = (args(5).toBoolean, args(6).toBoolean, args(1).toInt), n_iters = args(2).toInt,
      eta = args(3).toDouble, lambda = args(4).toDouble, normalization = args(10).toBoolean, args(11).toBoolean, 
      "adagrad", initWeights, valid_data, miniBatchFraction = args(12).toDouble)

    val test_predictionAndLabels = test_data.map(x => (ffm.predict(x._2), x._1))
    val test_metrics = new BinaryClassificationMetrics(test_predictionAndLabels)
    val test_auROC = test_metrics.areaUnderROC
    println("Test Area under ROC in day " + date.toString + " = " + test_auROC)
    ffm
  }

  override def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setAppName("TESTFFM"))

    if (args.length != 10) {
      println("testFFM <train_file> <k> <n_iters> <eta> <lambda> <k0> <k1> <partition_num> <valid_file> <test_file> <normal> <random> <miniBatchFraction>")
    }
    var line = ""
    for(arg <- args){
      line += arg + " "
    }
    println(line)
    var initWeights: Vector=null
    for (date <- 14 to 17){
      val ffm:FFMModel = train_one_day(sc, args, date, initWeights)
      initWeights = ffm.weights
    }

    val pre_path: String = "/user/gzsuweiyue/Data/netease_ctr/split/2017071[4567]/"
    val partition_num = args(7).toInt
    val test_data = load_ffm(sc, pre_path + args(9)).repartition(partition_num)
    val test_predictionAndLabels = test_data.map(x => (ffm.predict(x._2), x._1))
    val test_metrics = new BinaryClassificationMetrics(test_predictionAndLabels)
    val test_auROC = test_metrics.areaUnderROC
    println("Test Area under ROC in day " + date.toString + " = " + test_auROC)
    /*
    val partition_num = args(7).toInt
    val train_data = load_ffm(sc, args(0)).repartition(partition_num)
    val valid_data = load_ffm(sc, args(8)).repartition(partition_num)
    val test_data = load_ffm(sc, args(9)).repartition(partition_num)

    val data = train_data.union(valid_data)
    //sometimes the max feature/field number would be different in train_data/valid_data dataset,
    // so use the whole dataset to get the max feature/field number
    val m = data.flatMap(x=>x._2).map(_._1).collect.reduceLeft(_ max _) + 1
    val n = data.flatMap(x=>x._2).map(_._2).collect.reduceLeft(_ max _) + 1

    val ffm: FFMModel = FFMWithAdag.train(train_data, m, n, dim = (args(5).toBoolean, args(6).toBoolean, args(1).toInt), n_iters = args(2).toInt,
      eta = args(3).toDouble, lambda = args(4).toDouble, normalization = args(10).toBoolean, args(11).toBoolean, "adagrad", null, valid_data, miniBatchFraction = args(12).toDouble)

    val test_predictionAndLabels = test_data.map(x => (ffm.predict(x._2), x._1))
    val test_metrics = new BinaryClassificationMetrics(test_predictionAndLabels)
    val test_auROC = test_metrics.areaUnderROC
    println("Test Area under ROC = " + test_auROC)
    */

  }

}

