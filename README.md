# TensorFlowOnSpark

This code works over [TensorFlowOnSpark](https://github.com/yahoo/TensorFlowOnSpark) which uses a fork of [spark-ec2](https://github.com/anfeng/spark-ec2)

## Up and running

Clone TensorFlowOnSpark:
```
git clone https://github.com/yahoo/TensorFlowOnSpark.git
```
AWS p2 instances needs to be inside a public subnet and vpc

Before running:

* Create vpc with public subnet
* Turn on DNS Hostnames option on vpc

Two scripts have been generated:

* runEc2Tensorflow.sh: Set up Spark cluster ,installs everything necessary. It must be configured by environment variables
* tfos.sh: Make actions over cluster, receives one parameter, can be "start", "stop" or "destroy"

## Experiment 1

Run Tensorflow MNIST over 2 and 3 slaves Spark cluster with GPU computation.

* MNIST training
* Softmax model
* p2.xlarge instances for slaves. 1 GPU per machine
* m3.medium instance for master
* No RDMA
* Async training
* Hidden units: 128
* Batch size: 100

### Results

| Slaves        | Steps         | Precision  | Price   |   Training time   |
|:-------------:|:-------------:| :-----:    |:-------:|:-----------------:|
| 2             | 1000          | 97%        | 0,805 $ |  0:01:04          |
| 2             | 5000          | 98%        | 0,805 $ |  0:10:27          |
| 3             | 1000          | 96%        | 1,1 $   |  0:03:23          |
| 3             | 5000          | 98%        | 1,1 $   |  0:05:59          |

### Steps

1. Run cluster
```
cd PocTensorFlowOnSpark
sh runEc2Tensorflow.sh
export EC2_PEM_FILE=path/to/.pem
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ${EC2_PEM_FILE} root@<SPARK_MASTER_HOST>
```

2. Convert MNIST files into TensorFlow Record format

```
pushd ${TFoS_HOME}
spark-submit --master local[4] \
--jars ${TFoS_HOME}/tensorflow-hadoop-1.0-SNAPSHOT.jar \
--conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda/lib64" \
--driver-library-path="/usr/local/cuda/lib64" \
${TFoS_HOME}/examples/mnist/mnist_data_setup.py \
--output mnist/tfr \
--format tfr
popd
hadoop fs -ls mnist/tfr
```

3. Train a MNIST model
```
pushd ${TFoS_HOME}/src
zip -r ${TFoS_HOME}/tfspark.zip *
popd

export NUM_GPU=1
export CORES_PER_WORKER=4

export SPARK_WORKER_INSTANCES=2
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
export MASTER=spark://$(hostname):7077

spark-submit --master ${MASTER} \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--py-files ${TFoS_HOME}/tfspark.zip,${TFoS_HOME}/examples/mnist/tf/mnist_dist.py \
--conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda/lib64:$JAVA_HOME/jre/lib/amd64/server:$HADOOP_HOME/lib/native" \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--conf spark.executorEnv.HADOOP_HDFS_HOME=${HADOOP_HOME} \
--conf spark.executorEnv.CLASSPATH=$($HADOOP_HOME/bin/hadoop classpath --glob) \
--driver-library-path="/usr/local/cuda/lib64" \
${TFoS_HOME}/examples/mnist/tf/mnist_spark.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--images mnist/tfr/train --format tfr \
--steps 5000 \
--mode train --model mnist_model --tensorboard
```

The trained model and its check points should be located at HDFS.
```
hadoop fs -ls  /user/root/mnist_model
```

4. Destroy cluster

From your machine:
```
cd PocTensorFlowOnSpark
sh tfos.sh destroy
```

## Experiment 2

Run TFSlim over 2 and 3 slaves Spark cluster with GPU computation

*FAILED* experiment. Problems with HDFS support when reading data from Slim examples, they are not prepared to do it


1. Run cluster
```
cd PocTensorFlowOnSpark
sh runEc2Tensorflow.sh
export EC2_PEM_FILE=path/to/.pem
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i ${EC2_PEM_FILE} root@<SPARK_MASTER_HOST>
```

2. Convert MNIST files into TensorFlow Record format

```
pushd ${TFoS_HOME}
spark-submit --master local[4] \
--jars ${TFoS_HOME}/tensorflow-hadoop-1.0-SNAPSHOT.jar \
--conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda/lib64" \
--driver-library-path="/usr/local/cuda/lib64" \
${TFoS_HOME}/examples/mnist/mnist_data_setup.py \
--output mnist/tfr \
--format tfr
popd
hadoop fs -ls mnist/tfr
```

3. Update example to work with Tensorflow 0.12, is prepared to run over 0.11 version

* lenet_preprocessing.py line 42
```
nano ${TFoS_HOME}/examples/slim/preprocessing/lenet_preprocessing.py
```

Replace (TF 0.11)
```
image = tf.sub(image, 128.0)
```
By (TF 0.12)
```
image = tf.subtract(image, 128.0)
```

* download_and_convert_mnist.py line 168
```
nano ${TFoS_HOME}/examples/slim/datasets/download_and_convert_mnist.py
```

Replace (TF 0.11)
```
size = f.Size()
```
by (TF 0.12)
```
size = f.size()
```

4. Package the code as a Python zip/module

```
pushd ${TFoS_HOME}/src
zip -r ${TFoS_HOME}/tfspark.zip *
popd

pushd ${TFoS_HOME}/examples/slim;
zip -r ${TFoS_HOME}/slim.zip .;
popd
```

4. Train MNIST
```
export NUM_GPU=1
export DATASET_DIR=hdfs://user/root/mnist/tfr
export CORES_PER_WORKER=4
export SPARK_WORKER_INSTANCES=2
export TOTAL_CORES=$((${CORES_PER_WORKER} * ${SPARK_WORKER_INSTANCES}))
export MASTER=spark://$(hostname):7077
export QUEUE=gpu
export SPARK_EXECUTOR_INSTANCES=2


${SPARK_HOME}/bin/spark-submit --master ${MASTER} \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--conf spark.executor.instances=${SPARK_EXECUTOR_INSTANCES} \
--num-executors 2 \
--py-files ${TFoS_HOME}/tfspark.zip,${TFoS_HOME}/slim.zip \
--conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda/lib64:$JAVA_HOME/jre/lib/amd64/server:$HADOOP_HOME/lib/native" \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--conf spark.executorEnv.HADOOP_HDFS_HOME=${HADOOP_HOME} \
--conf spark.executorEnv.CLASSPATH=$($HADOOP_HOME/bin/hadoop classpath --glob) \
--driver-library-path="/usr/local/cuda/lib64" \
${TFoS_HOME}/examples/slim/train_image_classifier.py \
--dataset_dir ${DATASET_DIR} \
--dataset_name mnist \
--dataset_split_name train \
--model_name lenet \
--max_number_of_steps 1000 \
--num_gpus ${NUM_GPU} \
--batch_size 50 \
--tensorboard \
--num_ps_tasks 1
```

5. Destroy cluster

```
cd PocTensorFlowOnSpark
sh tfos.sh destroy
```

##  Good to know

### CUDA drivers problem

Recommended Yahoo's ami ami-f6d25596 updates nvidia library to 375 and generates the next problem after update:
```
Failed to initialize NVML: Driver/library version mismatch.
```
A new ami has been created in Oregon region with nvidia library updated, we recommend to use it

 - AMI Name: TensorflowOnSpark-0.12
 - AMI ID: ami-7ab1241a.

### Others

Output logs can be found in every slave on spark/work/

TensorBoard port change every execution, you may need to change security group rules to access.

The limit of AWS p2 instances per region is 1, you may need to request an update
