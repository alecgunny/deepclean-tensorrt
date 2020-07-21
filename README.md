View the plotted data at [nbviewer](https://nbviewer.jupyter.org/github/alecgunny/deepclean-tensorrt/blob/master/plot_throughput_latency.ipynb).

## Container build
```
TAG=20.06
docker build -t $USER/pytorch-tritonclient:$TAG --build-arg tag=$TAG .
```

## Build and populate model repository
```
docker run --rm -it -v $PWD:/workspace --gpus 1 -u $(id -u):$(id -g) \
  $USER/pytorch-tritonclient:$TAG python main.py
```

## Start server
```
docker run --rm -it -v $PWD/modelstore:/modelstore --gpus 1 -u $(id -u):$(id -g) \
  --name tritonserver nvcr.io/nvidia/tritonserver:20.06-py3 bin/tritonserver \
    --model-store /modelstore --model-control-mode=poll --repository-poll-secs 5
```

## Gather client data
```
./run_client 10
```

## Stop server
```
docker stop tritonserver
```
