package com.github.weiyuesu.tools

import scala.util.Random
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object FFMLoader extends Serializable {
  def load_ffm(sc: SparkContext, data_path: String, doBalance: Boolean=false, doShuffle: Boolean=false, balanceRatio: (Int, Int)=(1, 1)): RDD[(Double, Array[(Int, Int, Double)])] = {
    var data = sc.textFile(data_path).map(_.split(" ")).map(x => {
        val y = if(x(0).toInt > 0 ) 1.0 else -1.0
        val nodeArray: Array[(Int, Int, Double)] = x.drop(1).map(_.split(":")).map(x => {
            (x(0).toInt, x(1).toInt, x(2).toDouble)
          })
        (y, nodeArray)
      })
    if (doBalance){
      data = balance(data, balanceRatio)
    }
    if (doShuffle){
      data = shuffle(data)
    }
    data
  }

  def balance(data: RDD[(Double, Array[(Int, Int, Double)])], balanceRatio: (Int, Int)=(1, 1)): RDD[(Double, Array[(Int, Int, Double)])] = {
    data.flatMap(x => {
        val num: Int = if (x._1 == 1.0) balanceRatio._1 else balanceRatio._2
        Array.fill[(Double, Array[(Int, Int, Double)])](num)(x)
      })
  }

  def shuffle(data: RDD[(Double, Array[(Int, Int, Double)])]): RDD[(Double, Array[(Int, Int, Double)])] = {
    data.map(x => (new Random().nextDouble(), x)).sortByKey().map(x => x._2)
  }
}
