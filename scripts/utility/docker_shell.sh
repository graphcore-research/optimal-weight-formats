sudo docker run --rm -it --gpus all \
    -v "/home/ubuntu/.cache:/root/.cache" \
    -v /home/ubuntu/weight-formats:/home/ubuntu/weight-formats \
    -v $(pwd):$(pwd) -w $(pwd) \
    -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
    -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
    pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime "$@"
