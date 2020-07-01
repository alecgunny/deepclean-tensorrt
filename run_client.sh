#!/bin/bash -e

NUM_ITERS=$1
MODELS=(deepclean_onnx deepclean_trt_fp32 deepclean_trt_fp16)

if [[ ! -d logs ]]; then mkdir logs; fi

for i in $(seq 1 $NUM_ITERS); do
  for model in ${MODELS[@]}; do
    docker run --rm -it -v $PWD/logs:/logs -u $(id -u):$(id -g) --network container:tritonserver \
      nvcr.io/nvidia/tritonserver:20.06-py3-clientsdk install/bin/perf_client \
        -i grpc -m $model --concurrency-range 1:8:1 -f /logs/$model-1-$i.csv
    sed -i '/  count: 1/c\  count: 2' modelstore/$model/config.pbtxt
  done

  for model in ${MODELS[@]}; do
    docker run --rm -it -v $PWD/logs:/logs -u $(id -u):$(id -g) --network container:tritonserver \
      nvcr.io/nvidia/tritonserver:20.06-py3-clientsdk install/bin/perf_client \
        -i grpc -m $model --concurrency-range 1:10:1 -f /logs/$model-2-$i.csv
    sed -i '/  count: 2/c\  count: 4' modelstore/$model/config.pbtxt
  done

  for model in ${MODELS[@]}; do
    docker run --rm -it -v $PWD/logs:/logs -u $(id -u):$(id -g) --network container:tritonserver \
      nvcr.io/nvidia/tritonserver:20.06-py3-clientsdk install/bin/perf_client \
        -i grpc -m $model --concurrency-range 2:12:1 -f /logs/$model-4-$i.csv
    sed -i '/  count: 4/c\  count: 1' modelstore/$model/config.pbtxt
  done
done