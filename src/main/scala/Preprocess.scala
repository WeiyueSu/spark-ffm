import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import com.github.weiyuesu.tools.FFMLoader
import com.github.weiyuesu.tools.FeatureParser
import scala.util.Random

object Preprocess extends App {
  def ffm2node(line: String, dropTime: Boolean=false, setOne: Boolean=true): (Double, Array[(Int, Int, Double)]) = {
    val cols = line.split(" ")
    val y = if(cols(0).toInt > 0 ) 1.0 else -1.0
    val features = if (dropTime) cols.slice(1, cols.size - 1) else cols.slice(1, cols.size)
    val nodeArray: Array[(Int, Int, Double)] = features.map(x => x.split(":")).map(x => {
        if (setOne){
          (x(0).toInt, x(1).toInt, 1.0)
        }else{
          (x(0).toInt, x(1).toInt, x(2).toDouble)
        } 
      })
    (y, nodeArray)
  }

  override def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("Preprocess"))

    //val modelPath = "/user/gzsuweiyue/Model/ffm_adag_train_setOne_dynamicSample_convert_80train"
    //val resultPath = "/user/gzsuweiyue/Result/80train"

    //val ffm = FFMModel.load(sc, modelPath)

    val d = sc.parallelize(Array(1))
    val fieldPath = "/user/gzsuweiyue/Data/netease_ctr/scala/map/fieldMap"
    val featurePath = "/user/gzsuweiyue/Data/netease_ctr/scala/map/featureMap"
    val featureParser = new FeatureParser(d, fieldPath, featurePath)

    val dataPath = "/user/gzsuweiyue/Data/netease_ctr/txt/last_data/[0123456789]/"
    val ffmData = sc.textFile(dataPath)
    val udid2ffm = ffmData.map(featureParser.txt2ffm_line)

    udid2ffm.map(x => x._1 + "\t" + x._2).saveAsTextFile("/user/gzsuweiyue/Data/netease_ctr/txt/last_data_udid2ffm_20100")

    /*
    val udid2ffm = sc.textFile("/user/gzsuweiyue/Data/netease_ctr/txt/last_data_udid2ffm").map(x => x.split("\t")).map(x => (x(0), x(1)))

    val resultData = udid2ffm.map(x => (x._1, ffm2node(x._2))).map(x => (x._1, ffm.predict(x._2._2), x._2._1)) //udid, predict. label

    resultData.map(x => (x._2, (x._1, x._3))).sortByKey(false).map(x => "%s\t%s\t%s".format(x._2._1, x._1, x._2._2)).saveAsTextFile(resultPath)
    */

    //test_data.map(x => ffm.predict(x._2)).saveAsTextFile("/user/gzsuweiyue/Data/netease_ctr/ffm/all_test_result.txt")


    //val test_predictionAndLabels = test_data.map(x => (ffm.predict(x._2), x._1))
    //val test_metrics = new BinaryClassificationMetrics(test_predictionAndLabels)
    //val test_auROC = test_metrics.areaUnderROC
    //println("Test Area under ROC in day all = " + test_auROC)
  }
}
