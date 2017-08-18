package com.github.weiyuesu.tools

import scala.util.Random
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object FFMLoader extends Serializable {
  def load_ffm(sc: SparkContext, data_path: String, doBalance: Boolean=false, doShuffle: Boolean=false, balanceRatio: (Double, Double)=(1.0, 1.0), setOne: Boolean=false, dropTime: Boolean=false): RDD[(Double, Array[(Int, Int, Double)])] = {
    var data = sc.textFile(data_path).map(_.split(" ")).map(x => {
        val y = if(x(0).toInt > 0 ) 1.0 else -1.0
        val features = if (dropTime) x.slice(1, x.size - 1) else x.slice(1, x.size)
        val nodeArray: Array[(Int, Int, Double)] = features.map(_.split(":")).map(x => {
            if (setOne){
              (x(0).toInt, x(1).toInt, 1.0)
            }else{
              (x(0).toInt, x(1).toInt, x(2).toDouble)
            } 
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

  def balance(data: RDD[(Double, Array[(Int, Int, Double)])], balanceRatio: (Double, Double)=(1.0, 1.0)): RDD[(Double, Array[(Int, Int, Double)])] = {
    data.flatMap(x => {
        val ratio: Double = if (x._1 == 1.0) balanceRatio._1 else balanceRatio._2
        if(ratio >= 1){
          Array.fill[(Double, Array[(Int, Int, Double)])](ratio.toInt)(x)
        }else if(new Random().nextDouble() < ratio){
          Array[(Double, Array[(Int, Int, Double)])](x)
        }else{
          new Array[(Double, Array[(Int, Int, Double)])](0)
        }
      })
  }

  def shuffle(data: RDD[(Double, Array[(Int, Int, Double)])]): RDD[(Double, Array[(Int, Int, Double)])] = {
    data.map(x => (new Random().nextDouble(), x)).sortByKey().map(x => x._2)
  }
}
