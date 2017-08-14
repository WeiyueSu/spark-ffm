import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import com.github.weiyuesu.tools.FFMLoader
import scala.util.Random
import scopt._

object PredictFFM extends App {

  override def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("PredictFFM"))
    val modelPath = "/user/gzsuweiyue/Model/ffm_adag_train_n100xp_balance"
    val ffm = FFMModel.load(sc, modelPath)

    val pre_path: String = "/user/gzsuweiyue/Data/netease_ctr/ffm/all_test.ffm/"
    val test_data = FFMLoader.load_ffm(sc, pre_path).repartition(16)
    val test_predictionAndLabels = test_data.map(x => (ffm.predict(x._2), x._1))
    val test_metrics = new BinaryClassificationMetrics(test_predictionAndLabels)
    val test_auROC = test_metrics.areaUnderROC
    println("Test Area under ROC in day all = " + test_auROC)
  }
}
