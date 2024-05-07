# silero-rs - A Rust wrapper for the Silero C++ library

## Build

To build this library onnxruntime binaries are needed, they can be found here under assets: https://github.com/microsoft/onnxruntime/releases

Choose the library for your OS and architecture but not the libraries that have training in the name. Extract the binaries, they can be found in the lib folder.

When building run these commands on one terminal line: 
```
ORT_STRATEGY=system ORT_LIB_LOCATION='location of binaries' cargo build 
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
