import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import scala.util.Random


object TestFFMOnce extends App {
  def load_ffm(sc: SparkContext, data_path: String, balance: Boolean=false): RDD[(Double, Array[(Int, Int, Double)])] = {
    val data = sc.textFile(data_path).map(_.split(" ")).map(x => {
        val y = if(x(0).toInt > 0 ) 1.0 else -1.0
        val nodeArray: Array[(Int, Int, Double)] = x.drop(1).map(_.split(":")).map(x => {
            (x(0).toInt, x(1).toInt, x(2).toDouble)
          })
        (y, nodeArray)
      })
    if (balance){
      shuffleAndBalance(data)
    }else{
      data
    }
  }
  def shuffleAndBalance(data: RDD[(Double, Array[(Int, Int, Double)])]): RDD[(Double, Array[(Int, Int, Double)])] = {
    val balanceData = data.flatMap(x => {
        //val num: Int = if (x._1 == 1.0) 3922 else 1
        val num = 1
        Array.fill[(Double, Array[(Int, Int, Double)])](num)(x)
      })
    balanceData.map(x => (new Random().nextDouble(), x)).sortByKey().map(x => x._2)
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
    var initWeights: Vector = null

    val m = 23
    val n = 7672
    val partition_num = args(7).toInt

    val train_data = load_ffm(sc, "/user/gzsuweiyue/Data/netease_ctr/ffm/train_n100xp_balance.ffm", true).repartition(partition_num)
    val valid_data = load_ffm(sc, "/user/gzsuweiyue/Data/netease_ctr/ffm/valid_n100xp_balance.ffm", true).repartition(partition_num)
    //val train_data = load_ffm(sc, "/user/gzsuweiyue/Data/netease_ctr/split/20170714/train_adasyn100x_shuf.ffm", true).repartition(partition_num)
    //val valid_data = load_ffm(sc, "/user/gzsuweiyue/Data/netease_ctr/split/20170714/valid_adasyn100x_shuf.ffm", true).repartition(partition_num)

    val ffm: FFMModel = FFMWithAdag.train(train_data, m, n, dim = (args(5).toBoolean, args(6).toBoolean, args(1).toInt), n_iters = args(2).toInt,
      eta = args(3).toDouble, lambda = args(4).toDouble, normalization = args(10).toBoolean, args(11).toBoolean, 
      "adagrad", initWeights, valid_data, miniBatchFraction = args(12).toDouble)

    val pre_path: String = "/user/gzsuweiyue/Data/netease_ctr/split/2017071[4567]/"
    val test_data = load_ffm(sc, pre_path + args(9)).repartition(partition_num)
    val test_predictionAndLabels = test_data.map(x => (ffm.predict(x._2), x._1))
    val test_metrics = new BinaryClassificationMetrics(test_predictionAndLabels)
    val test_auROC = test_metrics.areaUnderROC
    println("Test Area under ROC in day all = " + test_auROC)
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

