/*
* Licensed to the Apache Software Foundation (ASF) under one or more
* contributor license agreements.  See the NOTICE file distributed with
* this work for additional information regarding copyright ownership.
* The ASF licenses this file to You under the Apache License, Version 2.0
* (the "License"); you may not use this file except in compliance with
* the License.  You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

package org.apache.spark.mllib.classification

import java.io._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._
import org.apache.hadoop.fs.Path
import breeze.linalg.{DenseVector => BDV}

import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.optimization.Gradient

import scala.util.Random
import org.apache.spark.mllib.util.{DataValidators, Loader, Saveable}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SparkSession}

case class Data(weights: Vector, n: Int, m: Int, k: Int, k0: Boolean, k1: Boolean, normalization: Boolean, solver: String)
/**
* Created by vincent on 16-12-19.
*/
/**
*
* @param numFeatures number of features
* @param numFields number of fields
* @param dim A (Boolean,Boolean,Int) 3-Tuple stands for whether the global bias term should be used, whether the
*            one-way interactions should be used, and the number of factors that are used for pairwise
*            interactions, respectively.
* @param isNorm whether normalize data
* @param weights weights of FFMModel
* @param solver "true": parallelizedSGD, parallelizedAdaGrad would be used otherwise
*/
class FFMModel(
  numFeatures: Int,
  numFields: Int,
  dim: (Boolean, Boolean, Int),
  isNorm: Boolean, 
  initWeights: Vector,
  solver: String="adag" ) 
extends Serializable {

  private var n: Int = numFeatures
  //numFeatures
  private var m: Int = numFields
  //numFields
  private var k: Int = dim._3
  //numFactors
  private var k0 = dim._1
  private var k1 = dim._2
  private var normalization: Boolean = isNorm
  private var initMean: Double = 0
  private var initStd: Double = 0.01
  val weights: Vector = initWeights

  require(n > 0 && k > 0 && m > 0)

  def radomization(l: Int, rand: Boolean): Array[Int] = {
    val order = Array.fill(l)(0)
    for (i <- 0 to l - 1) {
      order(i) = i
    }
    if (rand) {
      val rand = new Random()
      for (i <- l - 1 to 1) {
        val tmp = order(i - 1)
        val index = rand.nextInt(i)
        order(i - 1) = order(index)
        order(index) = tmp
      }
    }
    return order
  }


  def save(sc: SparkContext, path: String) {
    val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
    val metadata = compact(render(("class" -> this.getClass.getName) ~ ("version" -> "1.0")))
    sc.parallelize(Seq(metadata), 1).saveAsTextFile(Loader.metadataPath(path))

    // Create Parquet data.
    val data = Data(weights, n, m, k, k0, k1, normalization, solver)
    spark.createDataFrame(Seq(data)).repartition(1).write.parquet(Loader.dataPath(path))
  }


  def predict(data: Array[(Int, Int, Double)]): Double = {

    val r = if(isNorm){
      1 / data.map(x => math.pow(x._3, 2)).sum
    }else{
      1.0
    }
    val sqrt_r = math.sqrt(r)

    var t = 0.0
    if (k0) {
      solver match {
        case "sgd" => t += weights(weights.size - 1)
        case "adag" => t += weights(weights.size - 2)
        case "ftrl" => t += weights(weights.size - 3)
      }
    }

    val (align0, align1) = solver match {
      case "sgd" => (k, m * k)
      case "adag" => (k * 2, m * k * 2)
      case "ftrl" => (k * 3, m * k * 3)
    }

    // j: feature, f: field, v: value
    val valueSize = data.size //feature length
    var i = 0
    var ii = 0
    val pos = n * align1
    // j: feature, f: field, v: value
    for (i <- 0 to valueSize - 1) {
      val (f1, j1, v1) = data(i)

      if(k1) t += weights(pos + j1) * v1 * sqrt_r

      if (j1 < n && f1 < m) {
        for (ii <- i + 1 to valueSize - 1) {
          val (f2, j2, v2) = data(ii)
          if (j2 < n && f2 < m) {
            val w1_index: Int = j1 * align1 + f2 * align0
            val w2_index: Int = j2 * align1 + f1 * align0
            val v: Double = v1 * v2 * r
            for (d <- 0 to k - 1) {
              t += weights(w1_index + d) * weights(w2_index + d) * v
            }
          }
        }
      }
    }
    t
  }
}


object FFMModel {

  def load(sc: SparkContext, path: String): FFMModel = {
    val dataPath = Loader.dataPath(path)
    val spark = SparkSession.builder().sparkContext(sc).getOrCreate()
    val dataRDD = spark.read.parquet(dataPath)
    val dataArray = dataRDD.select("weights", "n", "m", "k", "k0", "k1", "normalization", "solver").take(1)
    assert(dataArray.length == 1, s"Unable to load data from: $dataPath")
    val data = dataArray(0)
    assert(data.size == 8, s"Unable to load data from: $dataPath")
    val (weights, n, m, k, k0, k1, normalization, solver) = data match {
      case Row(weights: Vector, n: Int, m: Int, k: Int, k0: Boolean, k1: Boolean, normalization: Boolean, solver: String) =>
        (weights, n, m, k, k0, k1, normalization, solver)
    }
    val args = Data(weights, n, m, k, k0, k1, normalization, solver)
    new FFMModel(args.n, args.m, (args.k0, args.k1, args.k), args.normalization, args.weights, args.solver)
  }

}

class FFMGradient(m: Int, n: Int, dim: (Boolean, Boolean, Int), solver: String="adag", isNorm: Boolean = true, 
                  weightCol: (Double, Double)=(1.0, 1.0)) extends Gradient {

  private val k0 = dim._1
  private val k1 = dim._2
  private val k = dim._3

  private def predict (data: Array[(Int, Int, Double)], weights: Vector, r: Double): Double = {

    val sqrt_r = math.sqrt(r)

    var t = 0.0
    if (k0) {
      solver match {
        case "sgd" => t += weights(weights.size - 1)
        case "adag" => t += weights(weights.size - 2)
        case "ftrl" => t += weights(weights.size - 3)
      }
    }

    val (align0, align1) = solver match {
      case "sgd" => (k, m * k)
      case "adag" => (k * 2, m * k * 2)
      case "ftrl" => (k * 3, m * k * 3)
    }

    val valueSize = data.size //feature length
    var i = 0
    var ii = 0
    val pos = n * align1
    // j: feature, f: field, v: value
    for (i <- 0 to valueSize - 1) {
      val (f1, j1, v1) = data(i)

      if(k1) t += weights(pos + j1) * v1 * sqrt_r

      if (j1 < n && f1 < m) {
        for (ii <- i + 1 to valueSize - 1) {

          val(f2, j2, v2) = data(ii)
          if (j2 < n && f2 < m) {
            val w1_index: Int = j1 * align1 + f2 * align0
            val w2_index: Int = j2 * align1 + f1 * align0
            val v: Double = v1 * v2 * r
            for (d <- 0 to k - 1) {
              t += weights(w1_index + d) * weights(w2_index + d) * v
            }
          }
        }
      }
    }
    t
  }

  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    throw new Exception("This part is merged into computeFFM()")
  }

  override def compute(data: Vector, label: Double, weights: Vector, cumGradient: Vector): Double = {
    throw new Exception("This part is merged into computeFFM()")
  }

  def computeFFM(
    label: Double, 
    data: Array[(Int, Int, Double)], 
    weights: Vector,
    eta: Double, 
    lambda: Double,
    do_update: Boolean, 
    solver: String="adag"): (BDV[Double], Double) = {

    val r = if(isNorm){
      1 / data.map(x => math.pow(x._3, 2)).sum
    }else{
      1.0
    }
    val sqrt_r = math.sqrt(r)
    val weightC = if(label == 1.0) weightCol._1 else weightCol._2

    val weightsArray: Array[Double] = weights.asInstanceOf[DenseVector].values
    var t = predict(data, weights, r)
    val expnyt = math.exp(-label * t)
    var tr_loss = if (expnyt.isInfinite){
      -label * t
    } else {
      math.log(1 + expnyt)
    }

    tr_loss *= weightC

    if(do_update){

      //val z = -label * t
      //val max_z = math.max(0, z)
      //val kappa = -label * math.exp(z - max_z) / (math.exp(z - max_z) + math.exp(0 - max_z))
      
      val kappa = -label * expnyt / (1 + expnyt) * weightC

      val (align0, align1) = solver match {
        case "sgd" => (k, m * k)
        case "adag" => (k * 2, m * k * 2)
        case "ftrl" => (k * 3, m * k * 3)
      }
      val valueSize = data.size //feature length
      var i = 0
      var ii = 0

      val r0, r1 = 0.0
      // parameter for ftrl
      val alpha, beta = eta
      val lambda1, lambda2 = lambda
      //val r0 = lambda 
      //val r1 = lambda
      val pos = n * align1

      // oldCode: weightsArray(weightsArray.length - 2) -= eta * (kappa + r0 * weightsArray(weightsArray.length - 2))
      if(k0) {
        solver match {
          case "sgd" => {
            val gk0: Double = kappa + r0 * weightsArray(weightsArray.length - 1)
            weightsArray(weightsArray.length - 1) -= eta * gk0
          }
          case "adag" => {
            val gk0: Double = kappa + r0 * weightsArray(weightsArray.length - 2)
            val wgk0: Double = weightsArray(weightsArray.length - 1) + gk0 * gk0
            weightsArray(weightsArray.length - 2) -= eta / (math.sqrt(wgk0)) * gk0
            weightsArray(weightsArray.length - 1) = wgk0
          }
          case "ftrl" => {
            val gk0: Double = kappa
            val wi = weightsArray.length - 3
            val zi = wi + 1
            val ni = zi + 1
            weightsArray(zi) += gk0 - weightsArray(wi) / alpha * (math.sqrt(weightsArray(ni) + math.pow(gk0, 2)) - math.sqrt(weightsArray(ni)))
            weightsArray(ni) += math.pow(gk0, 2)
            weightsArray(wi) = if (math.abs(weightsArray(zi)) <= lambda1) 0
                                else (math.signum(weights(zi)) * lambda1 - weights(zi)) / ((beta + math.sqrt(weightsArray(ni))) / alpha + lambda2)
          }
        }
      }

      for (i <- 0 to valueSize - 1) {
        val (f1, j1, v1) = data(i)
        if (j1 < n && f1 < m) {
          // oldCod: weightsArray(pos + j1) -= eta * (v1 * kappa * sqrt_r + r1 * weightsArray(pos + j1))
          if(k1) {
            solver match {
              case "sgd" => {
                val gk1 = v1 * kappa * sqrt_r + r1 * weightsArray(pos + j1)
                weightsArray(pos + j1) -= eta * gk1
              }
              case "adag" => {
                val gk1 = v1 * kappa * sqrt_r + r1 * weightsArray(pos + j1)
                val wgk1: Double = weightsArray(pos + j1 + n) + gk1 * gk1
                weightsArray(pos + j1) -= eta / (math.sqrt(wgk1)) * gk1
                weightsArray(pos + j1 + n) = wgk1
              }
              case "ftrl" => {
                val wi = pos + j1
                val zi = wi + n
                val ni = zi + n
                val gk1 = v1 * kappa * sqrt_r
                weightsArray(zi) += gk1 - weightsArray(wi) / alpha * (math.sqrt(weightsArray(ni) + math.pow(gk1, 2)) - math.sqrt(weightsArray(ni)))
                weightsArray(ni) += math.pow(gk1, 2)
                weightsArray(wi) = if (math.abs(weightsArray(zi)) <= lambda1) 0
                                    else (math.signum(weights(zi)) * lambda1 - weights(zi)) / ((beta + math.sqrt(weightsArray(ni))) / alpha + lambda2)
              }
            }
          }
          for (ii <- i + 1 to valueSize - 1) {
            val (f2, j2, v2) = data(ii)
            if (j2 < n && f2 < m) {
              val w1_index: Int = j1 * align1 + f2 * align0
              val w2_index: Int = j2 * align1 + f1 * align0
              val v: Double = v1 * v2 * r
              val wg1_index: Int = w1_index + k
              val wg2_index: Int = w2_index + k
              val kappav: Double = kappa * v
              for (d <- 0 to k - 1) {
                solver match {
                  case "sgd" => {
                    val g1: Double = lambda * weightsArray(w1_index + d) + kappav * weightsArray(w2_index + d)
                    val g2: Double = lambda * weightsArray(w2_index + d) + kappav * weightsArray(w1_index + d)
                    weightsArray(w1_index + d) -= eta * g1
                    weightsArray(w2_index + d) -= eta * g2
                  }
                  case "adag" => {
                    val g1: Double = lambda * weightsArray(w1_index + d) + kappav * weightsArray(w2_index + d)
                    val g2: Double = lambda * weightsArray(w2_index + d) + kappav * weightsArray(w1_index + d)
                    val wg1: Double = weightsArray(wg1_index + d) + g1 * g1
                    val wg2: Double = weightsArray(wg2_index + d) + g2 * g2
                    weightsArray(w1_index + d) -= eta / (math.sqrt(wg1)) * g1
                    weightsArray(w2_index + d) -= eta / (math.sqrt(wg2)) * g2
                    weightsArray(wg1_index + d) = wg1
                    weightsArray(wg2_index + d) = wg2
                  }
                  case "ftrl" => {
                    val g1: Double = kappav * weightsArray(w2_index + d)
                    val g2: Double = kappav * weightsArray(w1_index + d)
                    val wi1 = w1_index + d
                    val wi2 = w2_index + d
                    val zi1 = wi1 + k
                    val zi2 = wi2 + k
                    val ni1 = zi1 + k
                    val ni2 = zi2 + k
                    weightsArray(zi1) += g1 - weightsArray(wi1) / alpha * (math.sqrt(weightsArray(ni1) + math.pow(g1, 2)) - math.sqrt(weightsArray(ni1)))
                    weightsArray(ni1) += math.pow(g1, 2)
                    weightsArray(wi1) = (math.signum(weights(zi1)) * lambda1 - weights(zi1)) / ((beta + math.sqrt(weightsArray(ni1))) / alpha + lambda2)

                    weightsArray(zi2) += g2 - weightsArray(wi2) / alpha * (math.sqrt(weightsArray(ni2) + math.pow(g2, 2)) - math.sqrt(weightsArray(ni2)))
                    weightsArray(ni2) += math.pow(g2, 2)
                    weightsArray(wi2) = (math.signum(weights(zi2)) * lambda1 - weights(zi2)) / ((beta + math.sqrt(weightsArray(ni2))) / alpha + lambda2)
                  }
                }
              }
            }
          }
        }
      }
    }
    (BDV(weightsArray), tr_loss)
  }
}
