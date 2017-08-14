import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import com.github.weiyuesu.tools.FFMLoader
import scala.util.Random


object TestFFM extends App {

  def train_one_day(sc: SparkContext, args: Array[String], date: String, initWeights: Option[Vector], m: Int, n: Int): FFMModel = {

    //val pre_path: String = "/user/gzsuweiyue/Data/netease_ctr/scala/split/2017071" + date + "/"
    val pre_path: String = "/user/gzsuweiyue/Data/netease_ctr/split/2017071" + date + "/"
    val partition_num = args(7).toInt
    val train_data = FFMLoader.load_ffm(sc, pre_path + args(0), args(13).toDouble != args(14).toDouble, args(11).toBoolean, (args(13).toDouble, args(14).toDouble)).repartition(partition_num)
    println("train cnt: ", train_data.count())
    println("train positive", train_data.filter(x => x._1 == '1').count())
    val valid_data = FFMLoader.load_ffm(sc, pre_path + args(8), args(13).toDouble != args(14).toDouble, args(11).toBoolean, (args(13).toDouble, args(14).toDouble)).repartition(partition_num)
    val test_data = FFMLoader.load_ffm(sc, pre_path + args(9)).repartition(partition_num)

    val ffm: FFMModel = FFMWithAdag.train(train_data, m, n, dim=(args(5).toBoolean, args(6).toBoolean, args(1).toInt), n_iters=args(2).toInt,
      eta=args(3).toDouble, lambda=args(4).toDouble, normalization=args(10).toBoolean, 
      args(17), initWeights, Some(valid_data), miniBatchFraction=args(12).toDouble, redo=(args(15).toDouble, args(16).toDouble))

    val test_predictionAndLabels = test_data.map(x => (ffm.predict(x._2), x._1))
    val test_metrics = new BinaryClassificationMetrics(test_predictionAndLabels)
    val test_auROC = test_metrics.areaUnderROC
    println("Test Area under ROC in day " + date + " = " + test_auROC)
    ffm
  }

  override def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setAppName("TESTFFM"))

    println("testFFM <train_file> <k> <n_iters> <eta> <lambda> <k0> <k1> <partition_num> <valid_file> <test_file>" 
      + "<normal> <random> <miniBatchFraction> <balance1> <balance2> <redo1> <redo2> <solver> <model_path>")
    //val n = 7839
    val n = 7672
    val m = 23

    var line = ""
    for(arg <- args){
      line += arg + " "
    }
    println(line)

    var initWeights: Option[Vector] = None
    var ffm: FFMModel = null
    ffm = train_one_day(sc, args, "[4567]", initWeights, m, n)
    initWeights = Some(ffm.weights)

    //for (date <- 4 to 7){
      //ffm = train_one_day(sc, args, date.toString, initWeights, m, n)
      //initWeights = Some(ffm.weights)
    //}

    val pre_path: String = "/user/gzsuweiyue/Data/netease_ctr/scala/split/2017071[4567]/"
    val partition_num = args(7).toInt
    val test_data = FFMLoader.load_ffm(sc, pre_path + args(9)).repartition(partition_num)
    val test_predictionAndLabels = test_data.map(x => (ffm.predict(x._2), x._1))
    val test_metrics = new BinaryClassificationMetrics(test_predictionAndLabels)
    val test_auROC = test_metrics.areaUnderROC
    println("Test Area under ROC in day all = " + test_auROC)

  }

}

