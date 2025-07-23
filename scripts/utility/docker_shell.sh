docker run --rm -it --gpus all \
    -v "/home/ubuntu/.cache:/root/.cache" \
    -v /home/ubuntu/weight-formats:/home/ubuntu/weight-formats \
    -v $(pwd):$(pwd) -w $(pwd) \
    pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime "$@"
