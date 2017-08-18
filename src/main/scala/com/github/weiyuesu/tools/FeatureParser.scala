package com.github.weiyuesu.tools

import java.text.SimpleDateFormat
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Map
import org.apache.spark.{SparkConf, SparkContext}
import java.util.Date
import java.util.Calendar

class FeatureParser(data: RDD[Int], fieldPath: String, featurePath: String) extends Serializable {

  var featureMap: Map[String, (Int, Long, Long)] = if(isExists(featurePath)){
    val sc = data.context
    val featureRDD = sc.textFile(featurePath)
    val features = featureRDD.map(_.split("\t")).collect()
    var fMap: Map[String, (Int, Long, Long)] = Map()
    for(feature <- features){
      fMap(feature(0)) = (feature(1).toInt, feature(2).toLong, feature(3).toLong)
    }
    fMap
  }else{
    Map()
  }

  var fieldMap: Map[String, Int] = if(isExists(fieldPath)){
    val sc = data.context
    val fieldRDD = sc.textFile(fieldPath)
    val fields = fieldRDD.map(_.split("\t")).collect()
    var fMap: Map[String, Int] = Map()
    for(field <- fields){
      fMap += (field(0) -> field(1).toInt)
    }
    fMap
  }else{
    Map()
  }

  /*
  if(!fieldMap.contains("logTime")){
    fieldMap("logTime") = fieldMap.size
    for(hour <- 0 to 23; weekday <- 1 to 7){
      val feature: String = "t_" + weekday.toString + "_" + hour.toString
      featureMap(feature) = (featureMap.size, 1, 1)
    }
  }
  */

  var fieldMapBc = data.context.broadcast(fieldMap)
  var featureMapBc = data.context.broadcast(featureMap)

  def save(data: RDD[Int], fieldPath: String, featurePath: String): Unit = {
    val sc = data.context
    sc.parallelize(featureMap.map(x => x._1 + "\t" + x._2._1.toString + "\t" + x._2._2.toString + "\t" + x._2._3).toSeq).saveAsTextFile(featurePath)
    sc.parallelize(fieldMap.map(x => x._1 + "\t" + x._2.toString).toSeq).saveAsTextFile(fieldPath)
  }

  def updateFeatureMap(txts: RDD[String]): Unit = {
    val feat2feqRDD = txts.flatMap(l => {
      val cols = l.split("\t")
      val ffts: String = cols(1)
      val feature2freqArray = new ArrayBuffer[(String, Long)]
      for(fft <- ffts.split(" ")){
        val split_fft = fft.split(":")
        if (split_fft.size == 3){
          val feature = split_fft(0)
          if (isFeature(feature)){
            val field: String = feature2field(feature)
            val freq: Long = if (field == "01" && feature.size == 6) 1 else split_fft(1).toInt
            feature2freqArray += ((feature, freq))
          }
        }
      }
      feature2freqArray
    })

    val feat2cnt = feat2feqRDD.mapValues(x => 1).reduceByKey((x1, x2) => x1 + x2).collectAsMap()
    val feat2freq = feat2feqRDD.reduceByKey((x1, x2) => x1 + x2).collectAsMap()

    for (feature <- feat2freq.keys){
      if(featureMap.contains(feature)){
        featureMap(feature) = (featureMap(feature)._1, featureMap(feature)._2 + feat2cnt(feature), featureMap(feature)._3 + feat2freq(feature))
      }else{
        featureMap(feature) = (featureMap.size, feat2cnt(feature), feat2freq(feature))
        val field = feature2field(feature)
        if(!fieldMap.contains(field)){
          fieldMap(field) = fieldMap.size
        }
      }
    }
    featureMapBc = data.context.broadcast(featureMap)
    fieldMapBc = data.context.broadcast(fieldMap)
  }

  def txt2ffm(txts: RDD[String], update: Boolean=false): RDD[String] = {
    if(update){
      updateFeatureMap(txts)
    }
    txts.map(l => {
      val cols = l.split("\t")
      var line: String = cols(cols.size - 3)
      val ffts: String = cols(1)
      val logTime = cols(cols.size - 1)
      for(fft <- ffts.split(" ")){
        val split_fft = fft.split(":")
        if (split_fft.size == 3){
          val feature = split_fft(0)
          if (isFeature(feature)){
            val freq: Long = split_fft(1).toLong
            val lastTime: String = split_fft(2)
            val field = feature2field(feature)
            //val value = parseLastTime(lastTime, Some(logTime)) * parseFreq(feature, freq)
            val value = parseFreq(feature, freq)
            val fieldIdx = fieldMapBc.value(field)
            val featureIdx = featureMapBc.value(feature)._1
            line += " " + fieldIdx + ":" + featureIdx + ":" + value.toString
          }
        }
      }
      //line += " " + parseLogTime(logTime)
      line
    })
  }

  def txt2ffm_line(l: String): (String, String) = {
    val cols = l.split("\t")
    var line: String = cols(cols.size - 3)
    val ffts: String = cols(1)
    for(fft <- ffts.split(" ")){
      val split_fft = fft.split(":")
      if (split_fft.size == 3){
        var feature = split_fft(0)
        if (isFeature(feature)){
          val field = feature2field(feature)
          if (field == "640"){
            feature = "640_3"
          }else if(field == "00"){
            feature = "20100"
          }
          val value = 1.0
          val fieldIdx = fieldMapBc.value(field)
          val featureIdx = featureMapBc.value(feature)._1
          line += " " + fieldIdx + ":" + featureIdx + ":" + value.toString
        }
      }
    }
    (cols(0), line)
  }

  private def isExists(path: String): Boolean = {
    val conf = data.context.hadoopConfiguration
    val fs = org.apache.hadoop.fs.FileSystem.get(conf)
    fs.exists(new org.apache.hadoop.fs.Path(path))
  }

  private def isFeature(feature: String): Boolean = {
    val banFeatures: Array[String] = Array("809101", "1052701", "1061601", "1070801",
      "1080001", "1089201", "1098401", "1107601",
      "1116801", "1126001", "1135201", "1144401")
    !banFeatures.contains(feature) && feature.size != 2
  }


  private def parseLastTime(lastTime: String, nowTime: Option[String]): Double = {

    val nowTimeStamp: Long = if (nowTime.isEmpty) new Date().getTime() else new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").parse(nowTime.get).getTime()
    val lastTimeStamp: Long = new SimpleDateFormat("yyyyMMdd").parse(lastTime).getTime()

    val diffMs: Long = nowTimeStamp - lastTimeStamp

    val oneMonth: Long = 30.toLong * 24 * 60 * 60 * 1000
    val oneYear: Long = oneMonth * 12

    if (diffMs < oneMonth) 1.5 else if (diffMs < oneYear) 1.0 else 0.5
  }

  private def parseLogTime(logTime: String): String = {

    val logDate: Date = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").parse(logTime)
    val c: Calendar = Calendar.getInstance();
    c.setTime(logDate);
    val weekday = c.get(Calendar.DAY_OF_WEEK)
    val hour = logDate.getHours
    val field: String = "logTime"
    val feature: String = "t_" + weekday.toString + "_" + hour.toString
    fieldMap(field).toString + ":" + featureMap(feature)._1.toString + ":1"
  }


  private def feature2field(feature: String): String = {
    if (feature.contains("_")) feature.split("_")(0) else feature.slice(feature.size - 2, feature.size)
  }

  private def parseFreq(feature: String, featFreq: Long): Double = {
    val (idx, cnt, freq) = featureMapBc.value(feature)
    val aveFreq = freq.toDouble / cnt
    featFreq / aveFreq
  }
}

