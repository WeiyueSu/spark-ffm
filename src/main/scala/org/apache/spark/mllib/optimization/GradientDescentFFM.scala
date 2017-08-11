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
import scala.util.Random
import scala.collection.mutable.ArrayBuffer

/**
  * Created by vincent on 17-1-4.
  */
class GradientDescentFFM (private var gradient: Gradient, private var updater: Updater,
                          k: Int, n_iters: Int, eta: Double, lambda: Double,
                          normalization: Boolean) extends Optimizer {

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
               n_iters: Int, eta: Double, lambda: Double, solver: String, 
               validData: Option[RDD[(Double, Array[(Int, Int, Double)])]], redo: (Double, Double)=(1.0, 1.0)): Vector = {

    val (weights, _) = GradientDescentFFM.runMiniBatchAdag(data, gradient, initialWeights, n_iters, eta, lambda, solver, validData, miniBatchFraction, 0.0, redo)
    weights
  }


}

object GradientDescentFFM {
  def shuffle(data: RDD[(Double, Array[(Int, Int, Double)])]): RDD[(Double, Array[(Int, Int, Double)])] = {
    data.map(x => (new Random().nextDouble(), x)).sortByKey().map(x => x._2)
  }

  def runMiniBatchAdag(
                    trainData: RDD[(Double, Array[(Int, Int, Double)])],
                    gradient: Gradient,
                    initialWeights: Vector,
                    n_iters: Int,
                    eta: Double,
                    lambda: Double,
                    solver: String,
                    validData: Option[RDD[(Double, Array[(Int, Int, Double)])]],
                    miniBatchFraction: Double = 1.0,
                    convergenceTol: Double = 0.0, 
                    redo: (Double, Double)=(1.0, 1.0)) : (Vector, Array[Double]) = {
    val numIterations = n_iters
    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    var weights = initialWeights
    val n = weights.size
    val slices = trainData.getNumPartitions

    var converged = false // indicates whether converged based on convergenceTol
    var i = 0
    var minValidLoss: Double = Double.PositiveInfinity
    var bestWeights = initialWeights
    var stepCnt = 0

    val trainPosiData = trainData.filter(x => x._1 == 1.0)
    val trainNegaData = trainData.filter(x => x._1 != 1.0)

    val (validPosiData, validNegaData) = if (validData != None){
      (Some(validData.get.filter(x => x._1 == 1.0)), Some(validData.get.filter(x => x._1 != 1.0)))
    }else{
      (None, None)
    }

    while (!converged && i < numIterations && stepCnt < 2) {
      val bcWeights = trainData.context.broadcast(weights)
      // Sample a subset (fraction miniBatchFraction) of the total data
      val sampledTrainData = shuffle(trainPosiData.sample(redo._1 > 1.0, redo._1, i).union(trainNegaData.sample(redo._2 > 1.0, redo._2, i)))
      //val sampledTrainData = trainData.sample(false, miniBatchFraction, i)
      // compute and sum up the subgradients on this subset (this is one map-reduce)
      val (trainWSum, trainLossSum) = sampledTrainData.treeAggregate(BDV(bcWeights.value.toArray), 0.0)(
        seqOp = (c, v) => {
          val (w, loss) = gradient.asInstanceOf[FFMGradient].computeFFM(v._1, (v._2), Vectors.fromBreeze(c._1), eta, lambda, true, solver)
          (w, loss + c._2)
          /*
          val ratio: Double = if(v._1 == 1.0) redo._1 else redo._2
          if (ratio >= 1){
            val iters = ratio.toInt
            var w: BDV[Double] = c._1
            var loss: Double = c._2
            for(iter <- 1 to iters){
              val result = gradient.asInstanceOf[FFMGradient].computeFFM(v._1, (v._2), Vectors.fromBreeze(w), eta, lambda, true, solver)
              w = result._1
              loss += result._2
            }
            (w, loss)
          }else if(new Random().nextDouble() < ratio){
            val result = gradient.asInstanceOf[FFMGradient].computeFFM(v._1, (v._2), Vectors.fromBreeze(c._1), eta, lambda, true, solver)
            (result._1, c._2 + result._2)
          }else{
            c
          }
          */
        },
        combOp = (c1, c2) => {
          (c1._1 + c2._1, c1._2 + c2._2)
        }) // TODO: add depth level
      //val trainCnt = sampledTrainData.filter(x => x._1 == 1.0).count() * redo._1 + 
                      //sampledTrainData.filter(x => x._1 != 1.0).count() * redo._2  
      val trainCnt = sampledTrainData.count()
      i += 1
      weights = Vectors.dense(trainWSum.toArray.map(_ / slices))
      stochasticLossHistory += trainLossSum / trainCnt
      val trainLoss = trainLossSum / trainCnt

      if (validData == None){
        println("iter:" + i + ",tr_loss:" + trainLoss)
      }else{
        val sampledValidData = shuffle(validPosiData.get.sample(redo._1 > 1.0, redo._1, i).union(validNegaData.get.sample(redo._2 > 1.0, redo._2, i)))
        //val sampledValidData = validData.get.sample(false, miniBatchFraction, i)
        val (validWSum, validLossSum) = sampledValidData.treeAggregate(BDV(bcWeights.value.toArray), 0.0)(
          seqOp = (c, v) => {
            val (w, loss) = gradient.asInstanceOf[FFMGradient].computeFFM(v._1, (v._2), Vectors.fromBreeze(c._1), eta, lambda, false, solver)
            (w, loss + c._2)
            /*
            val ratio: Double = if(v._1 == 1.0) redo._1 else redo._2
            if (ratio >= 1){
              val iters = ratio.toInt
              var w: BDV[Double] = c._1
              var loss: Double = c._2
              for(iter <- 1 to iters){
                val result = gradient.asInstanceOf[FFMGradient].computeFFM(v._1, (v._2), Vectors.fromBreeze(w), eta, lambda, false, solver)
                w = result._1
                loss += result._2
              }
              (w, loss)
            }else if(new Random().nextDouble() < ratio){
              val result = gradient.asInstanceOf[FFMGradient].computeFFM(v._1, (v._2), Vectors.fromBreeze(c._1), eta, lambda, false, solver)
              (result._1, c._2 + result._2)
            }else{
              c
            }
            */
          },
          combOp = (c1, c2) => {
            (c1._1 + c2._1, c1._2 + c2._2)
          }) // TODO: add depth level

        val validCnt = sampledTrainData.count()
        //val validCnt = sampledValidData.filter(x => x._1 == 1.0).count() * redo._1 + 
                        //sampledValidData.filter(x => x._1 != 1.0).count() * redo._2  

        val validLoss = validLossSum / validCnt
        println("iter:" + i + ",tr_loss:" + trainLoss + ",va_loss:" + validLoss)

        if(validLoss < minValidLoss){
          minValidLoss = validLoss
          bestWeights = weights
          stepCnt = 0
        }else{
          stepCnt += 1
        }
      }
    }
    (bestWeights, stochasticLossHistory.toArray)
  }

}
