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

import breeze.linalg.{DenseVector => BDV}

import org.apache.spark.mllib.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.optimization.Gradient

import scala.util.Random

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
* @param n_iters number of iterations
* @param eta step size to be used for each iteration
* @param lambda regularization for pairwise interations
* @param isNorm whether normalize data
* @param random whether randomize data
* @param weights weights of FFMModel
* @param sgd "true": parallelizedSGD, parallelizedAdaGrad would be used otherwise
*/
class FFMModel(
  numFeatures: Int,
  numFields: Int,
  dim: (Boolean, Boolean, Int),
  n_iters: Int,
  eta: Double,
  lambda: Double,
  isNorm: Boolean, random: Boolean,
  initWeights: Vector,
  sgd: Boolean = true ) 
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

  def setOptimizer(op: String): Boolean = {
    if("sgd" == op) true else false
  }

  def predict(data: Array[(Int, Int, Double)]): Double = {

    val r = if(isNorm){
      1 / data.map(x => math.pow(x._3, 2)).sum
    }else{
      1.0
    }
    val sqrt_r = math.sqrt(r)

    var t = 0.0
    if (k0){
      if (sgd){
        t += weights(weights.size - 1)
      }else{
        t += weights(weights.size - 2)
      }
    }

    val (align0, align1) = if(sgd) {
      (k, m * k)
    } else {
      (k * 2, m * k * 2)
    }

    // j: feature, f: field, v: value
    val valueSize = data.size //feature length
    var i = 0
    var ii = 0
    val pos = if (sgd) n * m * k else n * m * k * 2
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

class FFMGradient(m: Int, n: Int, dim: (Boolean, Boolean, Int), sgd: Boolean = true, isNorm: Boolean = true) extends Gradient {

  private val k0 = dim._1
  private val k1 = dim._2
  private val k = dim._3

  private def predict (data: Array[(Int, Int, Double)], weights: Vector): Double = {

    val r = if(isNorm){
      1 / data.map(x => math.pow(x._3, 2)).sum
    }else{
      1.0
    }

    val sqrt_r = math.sqrt(r)

    var t = 0.0
    if (k0){
      if (sgd){
        t += weights(weights.size - 1)
      }else{
        t += weights(weights.size - 2)
      }
    }

    val (align0, align1) = if(sgd) {
      (k, m * k)
    } else {
      (k * 2, m * k * 2)
    }
    val valueSize = data.size //feature length
    var i = 0
    var ii = 0
    val pos = if (sgd) n * m * k else n * m * k * 2
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
    iter: Int, 
    solver: Boolean = true): (BDV[Double], Double) = {


    val r = if(isNorm){
      1 / data.map(x => math.pow(x._3, 2)).sum
    }else{
      1.0
    }
    val sqrt_r = math.sqrt(r)

    val weightsArray: Array[Double] = weights.asInstanceOf[DenseVector].values
    var t = predict(data, weights)
    val expnyt = math.exp(-label * t)
    var tr_loss = if (expnyt.isInfinite){
      -label * t
    } else {
      math.log(1 + expnyt)
    }

    //if (label == 1.0){
      //tr_loss /= 0.00025497016952045814
    //}

    if(do_update){
      //System.err.println("t: ", t, " label: ", label, " tr_loss: ", tr_loss, " expnyt: ", expnyt)

      //val z = -label * t
      //val max_z = math.max(0, z)
      //var kappa = -label * math.exp(z - max_z) / (math.exp(z - max_z) + math.exp(0 - max_z))

      //val kappa = -label * math.exp(z - max_z) / (math.exp(z - max_z) + math.exp(0 - max_z))
      val kappa = -label * expnyt / (1 + expnyt)

      val (align0, align1) = if (sgd) {
        (k, m * k)
      } else {
        (k * 2, m * k * 2)
      }
      val valueSize = data.size //feature length
      var i = 0
      var ii = 0

      //val r0, r1 = 0.00002
      val r0, r1 = 0.0
      var useOld = false
      //val r0 = lambda 
      //val r1 = lambda
      val pos = if (sgd) n * m * k else n * m * k * 2
      if (useOld){
        weightsArray(weightsArray.length - 2) -= eta * (kappa + r0 * weightsArray(weightsArray.length - 2))
      }else{
        if(k0) {
          if(sgd){
            val gk0: Double = kappa + r0 * weightsArray(weightsArray.length - 1)
            weightsArray(weightsArray.length - 1) -= eta * gk0
          }else{
            val gk0: Double = kappa + r0 * weightsArray(weightsArray.length - 2)
            val wgk0: Double = weightsArray(weightsArray.length - 1) + gk0 * gk0
            weightsArray(weightsArray.length - 2) -= eta / (math.sqrt(wgk0)) * gk0
            weightsArray(weightsArray.length - 1) = wgk0
          }
        }
      }

      for (i <- 0 to valueSize - 1) {
        val (f1, j1, v1) = data(i)
        if (j1 < n && f1 < m) {
          if(useOld){
            weightsArray(pos + j1) -= eta * (v1 * kappa * sqrt_r + r1 * weightsArray(pos + j1))
          }else{
            if(k1) {
              val gk1 = v1 * kappa * sqrt_r + r1 * weightsArray(pos + j1)
              if (sgd){
                weightsArray(pos + j1) -= eta * gk1
              }else{
                val wgk1: Double = weightsArray(pos + j1 + n) + gk1 * gk1
                weightsArray(pos + j1) -= eta / (math.sqrt(wgk1)) * gk1
                weightsArray(pos + j1 + n) = wgk1
                //System.err.println("v1: ", v1, "gk1: ", gk1, " wgk1: ", wgk1, " w: ", weightsArray(pos + j1),  " G: ", weightsArray(pos + j1 + n))
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
                val g1: Double = lambda * weightsArray(w1_index + d) + kappav * weightsArray(w2_index + d)
                val g2: Double = lambda * weightsArray(w2_index + d) + kappav * weightsArray(w1_index + d)
                if (sgd) {
                  weightsArray(w1_index + d) -= eta * g1
                  weightsArray(w2_index + d) -= eta * g2
                } else {
                  val wg1: Double = weightsArray(wg1_index + d) + g1 * g1
                  val wg2: Double = weightsArray(wg2_index + d) + g2 * g2
                  weightsArray(w1_index + d) -= eta / (math.sqrt(wg1)) * g1
                  weightsArray(w2_index + d) -= eta / (math.sqrt(wg2)) * g2
                  weightsArray(wg1_index + d) = wg1
                  weightsArray(wg2_index + d) = wg2

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

/*
var g_map = scala.collection.mutable.Map[Int, Double]()
// j: feature, f: field, v: value
// init g_map with lambda * w
for(i <- 0 to valueSize - 1) {
  val (f1, j1, v1) = data(i)
  if(k1) weightsArray(pos + j1) -= eta * (v1 * kappa + r1 * weightsArray(pos + j1))
    if (j1 < n && f1 < m) {
    for(ii <- i + 1 to valueSize - 1) {
      val (f2, j2, v2) = data(ii)
      if (j2 < n && f2 < m) {
        val w1_index: Int = j1 * align1 + f2 * align0
        val w2_index: Int = j2 * align1 + f1 * align0
        val v: Double = v1 * v2 * r
        val kappav: Double = kappa * v
        for (d <- 0 to k - 1) {
          if(!g_map.contains(w1_index + d)){
            g_map += (w1_index + d -> lambda * weightsArray(w1_index + d))
          }
          if(!g_map.contains(w2_index + d)){
            g_map += (w2_index + d -> lambda * weightsArray(w2_index + d))
          }
          g_map(w1_index + d) += kappav * weightsArray(w2_index + d)
          g_map(w2_index + d) += kappav * weightsArray(w1_index + d)
        }
      }
    }
  }
}
//
g_map.keys.foreach{ w_index => 
val wg_index: Int = w_index + k
val g: Double = g_map(w_index)
if (sgd) {
  weightsArray(w_index) -= eta * g
} else {
  val wg: Double = weightsArray(wg_index) + g * g
  weightsArray(wg_index) = wg
  weightsArray(w_index) -= eta / (math.sqrt(wg)) * g
}
  }
  */

