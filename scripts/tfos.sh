
export EC2_REGION=us-west-2
export EC2_ZONE=us-west-2b
export PROFILE=innovacion-dev-developer
export EC2_KEY=TFoSkeyPair
export EC2_PEM_FILE=/home/valentinquintas/.ssh/TFoSkeyPair.pem

export TFoS_HOME=/home/valentinquintas/TensorFlowOnSpark
export PROFILE=innovacion-dev-developer

export VPC_ID=vpc-689ae00f
export SUBNET_ID=subnet-b68035ff
export MASTER_INSTANCE_TYPE=m3.medium

action=$1

${TFoS_HOME}/scripts/spark-ec2 \
        --key-pair=${EC2_KEY} --identity-file=${EC2_PEM_FILE} \
        --region=${EC2_REGION} --zone=${EC2_ZONE} \
        --profile=${PROFILE} \
        --vpc-id ${VPC_ID} \
        --subnet-id ${SUBNET_ID} \
        ${action} TFoSdemo