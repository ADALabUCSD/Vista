/*
Copyright 2018 Supun Nakandala and Arun Kumar
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package vista.udf

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.Vectors

import scala.collection.mutable.WrappedArray

import javax.imageio.ImageIO
import java.awt.{Image, Color, Graphics2D}
import java.awt.image.{BufferedImage, DataBufferInt}
import java.io.ByteArrayInputStream
import java.nio.{ByteBuffer, ByteOrder, FloatBuffer}
import java.io.File
import java.util.ArrayList

import org.apache.spark.sql.DataFrame;
import org.apache.spark.api.java.JavaSparkContext
import org.apache.spark.sql.{SQLContext, Row}
import org.apache.spark.sql.types.{StructField, StringType, StructType, BinaryType, IntegerType, FloatType}
import org.apache.spark.input.PortableDataStream

import org.apache.spark.SparkFiles

/**
 * Contains helper functions implemented in Scala. These helper functions are in SparkSQL UDFs.
 */
object VistaUDFs {

    //To merge structures features and CNN features into one array
    def mergeFeatures(layer: Int, imageFeatures: Seq[Seq[Float]], structFeatures: Seq[Float], x:Int, y:Int, z:Int) = {
        var temp = imageFeatures(layer).toArray.map(_.toDouble)
        if (x > 1){
            //max pooling
            temp = maxPool(imageFeatures(layer).map(_.toDouble), x, y, z).flatten.flatten
        }
        Vectors.dense(temp ++ structFeatures.toArray.map(_.toDouble))
    }
    def mergeFeaturesUDF(): UserDefinedFunction = udf(mergeFeatures _)

    //To separate cnn features into corresponding layers in the bulk cnn inference method
    def sliceLayers(imageFeatures: Seq[Seq[Float]], cumSizes: Seq[Int]) : Array[Array[Float]] = {
        val x = imageFeatures(0).toArray
        val cumSizesArr = cumSizes.toArray
        val slicedX = new Array[Array[Float]](cumSizesArr.size - 1)

        for (i <- 0 to (cumSizesArr.size - 2)) {
            slicedX(i) = x.slice(cumSizesArr(i), cumSizesArr(i+1))
        }

        return slicedX
    }
    def sliceLayersUDF(): UserDefinedFunction = udf(sliceLayers _)

    def imageToByteArray(bytesArr: Array[Byte]) = {
        val image = ImageIO.read(new ByteArrayInputStream(bytesArr))
        val resizedImage = new BufferedImage(227, 227, BufferedImage.TYPE_INT_RGB)
        val g = resizedImage.createGraphics();
        g.drawImage(image, 0, 0, 227, 227, null);
        g.dispose();

        val pixelData = resizedImage.getRaster().getDataBuffer().asInstanceOf[DataBufferInt].getData()
        val rgbData = new Array[Float](pixelData.size*3)
        for (i <- 0 to 226) {
            for (j <- 0 to 226) {
               val c =  new Color(pixelData(227 * i + j))
               rgbData(227 * 3 * i + 3 * j) = c.getBlue().asInstanceOf[Float]
               rgbData(227 * 3 * i + 3 * j + 1) = c.getGreen().asInstanceOf[Float]
               rgbData(227 * 3 * i + 3 * j + 2) = c.getRed().asInstanceOf[Float]
            }
        }
        floatArrToBytes(rgbData)
    }
    def imageToByteArrayUDF(): UserDefinedFunction = udf(imageToByteArray _)

    def serializeCNNFeaturesArr(arr: Seq[Seq[Float]]) = floatArrToBytes(arr(0).toArray)
    def serializeCNNFeaturesArrUDF(): UserDefinedFunction = udf(serializeCNNFeaturesArr _)

    def getImagesDF(jsc: JavaSparkContext, dirPath: String): DataFrame = {
        val sc = JavaSparkContext.toSparkContext(jsc)
        val sqlContext = new SQLContext(sc)
        sqlContext.createDataFrame(sc.binaryFiles(dirPath).map(x => Row(getIdFromPath(x._1), x._2.toArray)),
            StructType(Array(StructField("id", StringType), StructField("image_buffer", BinaryType))))
    }

    def floatArrToBytes(arr: Array[Float]) = {
        val bbuf = ByteBuffer.allocate(4*arr.length)
        bbuf.order(ByteOrder.LITTLE_ENDIAN)
        bbuf.asFloatBuffer.put(FloatBuffer.wrap(arr))
        bbuf.array
    }

    //TODO configurable max/avg pooling by taking filter and strides as inputs
    def maxPool(imagesFeatures: Seq[Double], x: Int, y: Int, z: Int) = {
        val mp = Array.ofDim[Double](2, 2, z)
        var x1 = -1
        var x2 = -1
        var y1 = -1
        var y2 = -1

        //x and y are same (square conv filters)
        if (x%2 == 0){
            x1 = x/2 - 1
            y1 = x1
            x2 = x1 + 1
            y2 = x2
        }else{
            x1 = (x-1)/2
            y1 = x1
            x2 = x1
            y2 = x2
        }

        var max = Double.NegativeInfinity
        for (k <- 0 until z) {
            for (i <- 0.to(x1) ){
                max = Double.NegativeInfinity
                for (j <- 0.to(y1)){
                    if(imagesFeatures(k*x*y+i*y+j) > max){
                        max = imagesFeatures(k*x*y+i*y+j)
                    }
                }
                mp(0)(0)(k) = max

                max = Double.NegativeInfinity
                for (j <- y2.until(y)){
                    if(imagesFeatures(k*x*y+i*y+j) > max){
                        max = imagesFeatures(k*x*y+i*y+j)
                    }
                }
                mp(0)(1)(k) = max
            }

            for (i <- x2.until(x) ){
                max = Double.NegativeInfinity
                for (j <- 0.to(y1)){
                    if(imagesFeatures(k*x*y+i*y+j) > max){
                        max = imagesFeatures(k*x*y+i*y+j)
                    }
                }
                mp(1)(0)(k) = max

                max = Double.NegativeInfinity
                for (j <- y2.until(y)){
                    if(imagesFeatures(k*x*y+i*y+j) > max){
                        max = imagesFeatures(k*x*y+i*y+j)
                    }
                }
                mp(1)(1)(k) = max
            }
        }
        mp
    }

}