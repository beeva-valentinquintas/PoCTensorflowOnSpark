export AMI_IMAGE=ami
export EC2_REGION=us-west-2
export EC2_ZONE=us-west-2a
export SPARK_WORKER_INSTANCES=2
export EC2_INSTANCE_TYPE=p2.xlarge
export MASTER_INSTANCE_TYPE=m3.medium
export EC2_MAX_PRICE=0.6

export PROFILE=aws_profile
export TFoS_HOME=path_to_TensorflowOnSpark
export EC2_KEY=key_pair
export EC2_PEM_FILE=path_to_your_pem_file
export VPC_ID=a_vpc
export SUBNET_ID=a_subnet

${TFoS_HOME}/scripts/spark-ec2 \
        --key-pair=${EC2_KEY} --identity-file=${EC2_PEM_FILE} \
        --profile=${PROFILE} \
        --region=${EC2_REGION} \
        --zone=${EC2_ZONE} \
        --ebs-vol-size=50 \
        --instance-type=${EC2_INSTANCE_TYPE} \
        --ami=${AMI_IMAGE} -s ${SPARK_WORKER_INSTANCES} \
        --copy-aws-credentials \
        --hadoop-major-version=yarn --spark-version 1.6.0 \
        --no-ganglia \
        --user-data ${TFoS_HOME}/scripts/ec2-cloud-config.txt \
        --subnet-id ${SUBNET_ID} \
        --vpc-id ${VPC_ID} \
        --master-instance-type ${MASTER_INSTANCE_TYPE} \
        --spot-price ${EC2_MAX_PRICE} \
        launch TFoSdemo
