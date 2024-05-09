# silero-rs - A Rust wrapper for the Silero C++ library

## Build

Building this application requires the onnxruntime binaries, which can be found in the assets section here <https://github.com/microsoft/onnxruntime/releases>. When choosing which version to download, select the one that matches your OS but doesn’t have “training” within the name. The binaries can be found in the lib folder.

The following command can then be used to build the application: 
```
ORT_STRATEGY=system ORT_LIB_LOCATION=<path to binaries> cargo build 
```
## Silero Example

Run with
```
cargo run -- --model <MODEL> --audio <AUDIO>
```
or for the binary direct

```
silero_example --model <MODEL> --audio <AUDIO>
```

where <MODEL> is the path to the Silero model and <AUDIO> is the path to the audio 

## Silero Information

Silero Meta Data
Name: torch_jit
Produced by pytorch
Inputs:
0 input: Tensor<f32>(dyn, dyn)
1 sr: Tensor<i64>()
2 h: Tensor<f32>(2, dyn, 64)
3 c: Tensor<f32>(2, dyn, 64)
Outputs:
0 output: Tensor<f32>(dyn, 1)
1 hn: Tensor<f32>(2, dyn, 64)
2 cn: Tensor<f32>(2, dyn, 64)
