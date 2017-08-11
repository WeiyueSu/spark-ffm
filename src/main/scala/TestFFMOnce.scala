import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import com.github.weiyuesu.tools.FFMLoader
import scala.util.Random
import scopt._

object TestFFMOnce extends App {

  override def main(args: Array[String]): Unit = {

    val sc = new SparkContext(new SparkConf().setAppName("TESTFFMOnce"))

    println("testFFM <train_file> <k> <n_iters> <eta> <lambda> <k0> <k1> <partition_num> <valid_file> <test_file> "
      + "<normal> <random> <miniBatchFraction> <balance> <redo> <solver> <model_path>")

    var line = ""
    args.foreach{arg =>
      line += arg + " "
    }
    println(line)
    var initWeights: Option[Vector] = None

    val m = 23
    val n = 7672
    val partition_num = args(7).toInt

    val train_data = FFMLoader.load_ffm(sc, "/user/gzsuweiyue/Data/netease_ctr/ffm/train_n10xp_balance.ffm", doShuffle=args(11).toBoolean).repartition(partition_num)
    val valid_data = FFMLoader.load_ffm(sc, "/user/gzsuweiyue/Data/netease_ctr/ffm/valid_n10xp_balance.ffm", doShuffle=args(11).toBoolean).repartition(partition_num)

    var ffm: FFMModel = FFMWithAdag.train(train_data, m, n, dim = (args(5).toBoolean, args(6).toBoolean, args(1).toInt), n_iters = args(2).toInt,
      eta = args(3).toDouble, lambda = args(4).toDouble, normalization = args(10).toBoolean, 
      args(15), initWeights, Some(valid_data), miniBatchFraction = args(12).toDouble)

    //val modelPath = args(16)
    //ffm.save(sc, modelPath)
    //ffm = FFMModel.load(sc, modelPath)

    val pre_path: String = "/user/gzsuweiyue/Data/netease_ctr/split/2017071[4567]/"
    val test_data = FFMLoader.load_ffm(sc, pre_path + args(9)).repartition(partition_num)
    val test_predictionAndLabels = test_data.map(x => (ffm.predict(x._2), x._1))
    val test_metrics = new BinaryClassificationMetrics(test_predictionAndLabels)
    val test_auROC = test_metrics.areaUnderROC
    println("Test Area under ROC in day all = " + test_auROC)

  }

}

