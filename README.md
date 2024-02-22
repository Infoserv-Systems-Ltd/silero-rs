# silero-rs - A Rust wrapper for the Silero C++ library

## Development 

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
