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

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.optimization._

import scala.util.Random
/**
  * Created by vincent on 17-1-4.
  */
/**
  *
  * @param m number of fields of input data
  * @param n number of features of input data
  * @param dim A (Boolean,Boolean,Int) 3-Tuple stands for whether the global bias term should be used, whether the
  *            one-way interactions should be used, and the number of factors that are used for pairwise
  *            interactions, respectively.
  * @param n_iters number of iterations
  * @param eta step size to be used for each iteration
  * @param lambda regularization for pairwise interactions
  * @param normalization whether normalize data
  * @param solver "solver": parallelizedSGD, parallelizedAdaGrad would be used otherwise
  */
class FFMWithAdag(m: Int, n: Int, dim: (Boolean, Boolean, Int), n_iters: Int, eta: Double, lambda: Double,
                  normalization: Boolean, solver: String) extends Serializable {
  private val k0 = dim._1
  private val k1 = dim._2
  private val k = dim._3

  println("get numFields:" + m + ",nunFeatures:" + n + ",numFactors:" + k)
  private def generateInitWeights(): Vector = {
    val (num_k0, num_k1) = (k0, k1) match {
      case (true, true) =>
        (1 * n, 1)
      case(true, false) =>
        (1 * n, 0)
      case(false, true) =>
        (0, 1)
      case(false, false) =>
        (0, 0)
    }

    val tmpSize = solver match {
      case "sgd" => n * m * k + num_k1 + num_k0
      case "adag" => 2 * (n * m * k + num_k1 + num_k0)
      case "ftrl" => 3 * (n * m * k + num_k1 + num_k0)
    }

    println("allocating:" + tmpSize)
    val W = new Array[Double](tmpSize)

    val coef = 1.0 / Math.sqrt(k)
    val random = new Random()
    var position = 0
    solver match {
      case "sgd" => {
        for (j <- 0 to n - 1; f <- 0 to m - 1; d <- 0 to k - 1) {
          W(position) = coef * random.nextDouble()
          position += 1
        }
        if (k1) {
          for (p <- 0 to n - 1) {
            W(position) = 0.0
            position += 1
          }
        }
        if (k0) W(position) = 0.0
      }
      case "adag" => {
        for (j <- 0 to n - 1; f <- 0 to m - 1; d <- 0 to 2 * k - 1) {
          W(position) = if (d < k) coef * random.nextDouble() else 1.0
          position += 1
        }
        if (k1) {
          for (p <- 0 to 2 * n - 1) {
            W(position) = if (p < n) 0.0 else 1.0
            position += 1
          }
        }
        if (k0){
          W(position) = 0.0
          position += 1
          W(position) = 1.0
        }
      }
      case "ftrl" => {
        for (j <- 0 to n - 1; f <- 0 to m - 1; d <- 0 to 3 * k - 1) {
          W(position) = if (d < k) coef * random.nextDouble() else 1.0
          position += 1
        }
        if (k1) {
          for (p <- 0 to 3 * n - 1) {
            W(position) = 0.0
            position += 1
          }
        }
        if (k0){
          for (p <- 0 to 2) {
            W(position) = 0.0
            position += 1
          }
        }
      }
    }
    Vectors.dense(W)
  }

  /**
  * Create a FFMModle from an encoded vector.
  */
  private def createModel(weights: Vector): FFMModel = {
    //val values = weights.toArray
    new FFMModel(n, m, dim, normalization, weights, solver)
  }

  /**
  * Run the algorithm with the configured parameters on an input RDD
  * of FFMNode entries.
  */

  def run(input: RDD[(Double, Array[(Int, Int, Double)])], initWeights: Option[Vector], valid_data: Option[RDD[(Double, Array[(Int, Int, Double)])]], 
    miniBatchFraction: Double=1.0, redo: (Double, Double)=(1.0, 1.0), weightCol: (Double, Double)=(1.0, 1.0)): FFMModel = {
    
    val weights = if(initWeights == None){
      generateInitWeights()
    }else{
      initWeights.get
    }
    val gradient = new FFMGradient(m, n, dim, solver, normalization, weightCol)
    val optimizer = new GradientDescentFFM(gradient, null, k, n_iters, eta, lambda, normalization)
    optimizer.setMiniBatchFraction(miniBatchFraction)

    val new_weights = optimizer.optimize(input, weights, n_iters, eta, lambda, solver, valid_data, redo)
    createModel(new_weights)
  }

}

object FFMWithAdag {
  /**
  *
  * @param data input data RDD
  * @param m number of fields of input data
  * @param n number of features of input data
  * @param dim A (Boolean,Boolean,Int) 3-Tuple stands for whether the global bias term should be used, whether the
  *            one-way interactions should be used, and the number of factors that are used for pairwise
  *            interactions, respectively.
  * @param n_iters number of iterations
  * @param eta step size to be used for each iteration
  * @param lambda regularization for pairwise interactions
  * @param normalization whether normalize data
  * @param solver "solver": parallelizedSGD, parallelizedAdaGrad would be used otherwise
  * @return FFMModel
  */
  def train(data: RDD[(Double, Array[(Int, Int, Double)])], m: Int, n: Int,
    dim: (Boolean, Boolean, Int), n_iters: Int, eta: Double, lambda: Double, normalization: Boolean, 
    solver: String = "adag", initWeights: Option[Vector]=None, valid_data: Option[RDD[(Double, Array[(Int, Int, Double)])]]=None, 
    miniBatchFraction: Double=1.0, redo: (Double, Double)=(1.0, 1.0), weightCol: (Double, Double)=(1.0, 1.0)): FFMModel = {

    new FFMWithAdag(m, n, dim, n_iters, eta, lambda, normalization, solver)
    .run(data, initWeights, valid_data, miniBatchFraction, redo, weightCol)
  }
}
