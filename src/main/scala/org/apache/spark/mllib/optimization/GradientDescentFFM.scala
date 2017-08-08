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

package org.apache.spark.mllib.optimization

import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

/**
  * Created by vincent on 17-1-4.
  */
class GradientDescentFFM (private var gradient: Gradient, private var updater: Updater,
                          k: Int, n_iters: Int, eta: Double, lambda: Double,
                          normalization: Boolean, random: Boolean) extends Optimizer {

  val sgd = true
  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0
  private var convergenceTol: Double = 0.001
  /**
    * Set the initial step size of SGD for the first step. Default 1.0.
    * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
    */
  def setStepSize(step: Double): this.type = {
    require(step > 0,
      s"Initial step size must be positive but got ${step}")
    this.stepSize = step
    this
  }

  /**
    * :: Experimental ::
    * Set fraction of data to be used for each SGD iteration.
    * Default 1.0 (corresponding to deterministic/classical gradient descent)
    */
  @Experimental
  def setMiniBatchFraction(fraction: Double): this.type = {
    require(fraction > 0 && fraction <= 1.0,
      s"Fraction for mini-batch SGD must be in range (0, 1] but got ${fraction}")
    this.miniBatchFraction = fraction
    this
  }

  /**
    * Set the number of iterations for SGD. Default 100.
    */
  def setNumIterations(iters: Int): this.type = {
    require(iters >= 0,
      s"Number of iterations must be nonnegative but got ${iters}")
    this.numIterations = iters
    this
  }

  /**
    * Set the regularization parameter. Default 0.0.
    */
  def setRegParam(regParam: Double): this.type = {
    require(regParam >= 0,
      s"Regularization parameter must be nonnegative but got ${regParam}")
    this.regParam = regParam
    this
  }

  /**
    * Set the convergence tolerance. Default 0.001
    * convergenceTol is a condition which decides iteration termination.
    * The end of iteration is decided based on below logic.
    *
    *  - If the norm of the new solution vector is >1, the diff of solution vectors
    *    is compared to relative tolerance which means normalizing by the norm of
    *    the new solution vector.
    *  - If the norm of the new solution vector is <=1, the diff of solution vectors
    *    is compared to absolute tolerance which is not normalizing.
    *
    * Must be between 0.0 and 1.0 inclusively.
    */
  def setConvergenceTol(tolerance: Double): this.type = {
    require(tolerance >= 0.0 && tolerance <= 1.0,
      s"Convergence tolerance must be in range [0, 1] but got ${tolerance}")
    this.convergenceTol = tolerance
    this
  }

  /**
    * Set the gradient function (of the loss function of one single data example)
    * to be used for SGD.
    */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }


  /**
    * Set the updater function to actually perform a gradient step in a given direction.
    * The updater is responsible to perform the update from the regularization term as well,
    * and therefore determines what kind or regularization is used, if any.
    */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }
  def optimize(data: RDD[(Double, Vector)], initialWeights: Vector): Vector = {
    Array(1).toVector.asInstanceOf[Vector]

  }
  def optimize(data: RDD[(Double, Array[(Int, Int, Double)])], initialWeights: Vector,
               n_iters: Int, eta: Double, lambda: Double, solver: Boolean, valid_data: RDD[(Double, Array[(Int, Int, Double)])]): Vector = {
    val (weights, _) = GradientDescentFFM.runMiniBatchAdag(data, gradient, initialWeights, n_iters, eta, lambda, solver, valid_data: RDD[(Double, Array[(Int, Int, Double)])], miniBatchFraction)
    weights
  }


}

object GradientDescentFFM {
  def runMiniBatchAdag(
                    train_data: RDD[(Double, Array[(Int, Int, Double)])],
                    gradient: Gradient,
                    initialWeights: Vector,
                    n_iters: Int,
                    eta: Double,
                    lambda: Double,
                    solver: Boolean,
                    valid_data: RDD[(Double, Array[(Int, Int, Double)])],
                    miniBatchFraction: Double = 1.0,
                    convergenceTol: Double = 0.0) : (Vector, Array[Double]) = {
    val numIterations = n_iters
    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    var weights = initialWeights
    val n = weights.size
    val slices = train_data.getNumPartitions


    var converged = false // indicates whether converged based on convergenceTol
    var i = 0
    var minValidLoss: Double = Double.PositiveInfinity
    var bestWeights = initialWeights
    var breakFlag = false
    var stepCnt = 0

    while (!converged && i < numIterations && stepCnt < 2) {
      val bcWeights = train_data.context.broadcast(weights)
      // Sample a subset (fraction miniBatchFraction) of the total data
      val sampled_train_data = train_data.sample(false, miniBatchFraction, i)
      // compute and sum up the subgradients on this subset (this is one map-reduce)
      val (wSum, lSum) = sampled_train_data.treeAggregate(BDV(bcWeights.value.toArray), 0.0)(
        seqOp = (c, v) => {
          //val (w: BDV[Double], loss: Double) = gradient.asInstanceOf[FFMGradient].computeFFM(v._1, (v._2), Vectors.fromBreeze(c._1),
            //eta, lambda, true, i, solver)

          var w: BDV[Double] = c._1
          var loss: Double = 0.0
          //val iters: Int = if(v._1 == 1.0) (1 / 0.00025497016952045814).toInt else 1
          val iters: Int = 1
          for(i <- 1 to iters){
            val result = gradient.asInstanceOf[FFMGradient].computeFFM(v._1, (v._2), Vectors.fromBreeze(w), eta, lambda, true, i, solver)
            w = result._1
            loss = result._2
          }
          (w, loss + c._2)
        },
        combOp = (c1, c2) => {
          (c1._1 + c2._1, c1._2 + c2._2)
        }) // TODO: add depth level
      val train_cnt = sampled_train_data.count()

      i += 1
      weights = Vectors.dense(wSum.toArray.map(_ / slices))
      stochasticLossHistory += lSum / slices
      //println("iter:" + i + ",tr_loss:" + lSum / slices)

      val sampled_valid_data = valid_data.sample(false, miniBatchFraction, i)
      val (valid_wSum, valid_lSum) = sampled_valid_data.treeAggregate(BDV(bcWeights.value.toArray), 0.0)(
        seqOp = (c, v) => {
          val (w: BDV[Double], loss: Double) = gradient.asInstanceOf[FFMGradient].computeFFM(v._1, (v._2), Vectors.fromBreeze(c._1),
            eta, lambda, false, i, solver)
          (w, loss + c._2)
        },
        combOp = (c1, c2) => {
          (c1._1 + c2._1, c1._2 + c2._2)
        }) // TODO: add depth level
      val valid_cnt = sampled_valid_data.count()
      val valid_loss = valid_lSum / valid_cnt
      println("iter:" + i + ",tr_loss:" + lSum / train_cnt + ",va_loss:" + valid_loss)

      if(valid_loss < minValidLoss){
        minValidLoss = valid_loss
        bestWeights = weights
        stepCnt = 0
      }else{
        //weights = bestWeights
        //breakFlag = true
        stepCnt += 1
      }
    }
    (bestWeights, stochasticLossHistory.toArray)
  }

  /*
  def parallelAdag(
    data: RDD[(Double, Array[(Int, Int, Double)])],
    gradient: Gradient,
    initialWeights: Vector,
    n_iters: Int,
    eta: Double,
    lambda: Double,
    solver: Boolean,
    valid_data: RDD[(Double, Array[(Int, Int, Double)])]) : (Vector, Array[Double]) = {
    val numIterations = n_iters
    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    var weights = Vectors.dense(initialWeights.toArray)
    val n = weights.size
    val slices = data.getNumPartitions


    var converged = false // indicates whether converged based on convergenceTol
    var i = 0
    while (!converged && i < numIterations) {
      val bcWeights = data.context.broadcast(weights)
      // Sample a subset (fraction miniBatchFraction) of the total data
      // compute and sum up the subgradients on this subset (this is one map-reduce)

      val (wSum, lSum) = data.treeAggregate(BDV(bcWeights.value.toArray), 0.0)(
        seqOp = (c, v) => {
          gradient.asInstanceOf[FFMGradient].computeFFM(v._1, (v._2), Vectors.fromBreeze(c._1),
            eta, lambda, true, i, solver)
        },
        combOp = (c1, c2) => {
          (c1._1 + c2._1, c1._2 + c2._2)
        }) // TODO: add depth level

      i += 1
      weights = Vectors.dense(wSum.toArray.map(_ / slices))
      stochasticLossHistory += lSum / slices
      //println("iter:" + i + ",tr_loss:" + lSum / slices)

      val (valid_wSum, valid_lSum) = valid_data.treeAggregate(BDV(bcWeights.value.toArray), 0.0)(
        seqOp = (c, v) => {
          gradient.asInstanceOf[FFMGradient].computeFFM(v._1, (v._2), Vectors.fromBreeze(c._1),
            eta, lambda, false, i, solver)
        },
        combOp = (c1, c2) => {
          (c1._1 + c2._1, c1._2 + c2._2)
        }) // TODO: add depth level
      println("iter:" + i + ",tr_loss:" + lSum / slices + ",va_loss:" + valid_lSum / slices)
    }

    (weights, stochasticLossHistory.toArray)

  }
  */

}
