# silero-rs - A Rust wrapper for the Silero C++ library

![GitHub Actions Badge](https://github.com/Infoserv-Systems-Ltd/silero-rs/actions/workflows/rust.yml/badge.svg)

This library provides a wrapper to use Silero voice activation detection in rust. Silero is a neural network that can detect whether recorded audio contains human speech. It does this by being fed small sections of the audio and returning a probability for each section as to whether it contains human speech. This library works directly with WAV files, any other file type must first have the audio data extracted. Silero is run using onnxruntime, this is possible due to the onnxruntime_rs crate that provides a wrapper in rust for onnxruntime.

## Usage

Below is a basic example showing how to create a VAD session, load audio data, and run voice detection. For more advanced examples, see the "Detection Modes" section below.

```rust
use std::path::PathBuf;
use vad_rust::{VadSession, VoiceDetectResult};

fn main() {
    // Load the VAD model from a file
    let model_path = PathBuf::from("path/to/silero_vad.onnx");
    let mut vad_session = VadSession::new(&model_path, 0.5).unwrap();

    // Load WAV file and run voice detection
    let results: Vec<VoiceDetectResult> = vad_session.run_voice_detection_wav("audio_file.wav").unwrap();

    // Output detection results
    for result in results {
        println!("Detected voice: {}, Probability: {}", result.voice_detect, result.probability);
    }
}
```
You can pass additional options to configure the VAD session or adjust the detection threshold to fine-tune the accuracy of voice detection.

## Detection Modes

The library supports both stateless and stateful voice detection:

1. Stateless Detection: Processes audio data in windows but does not maintain any internal state across windows.

```rust
let results = vad_session.run_voice_detection(audio_data, 16000)?;
```

2. Stateful Detection: Maintains state (such as hidden vectors) across multiple windows, making it ideal for continuous audio streams.

```rust
let results = vad_session.run_stateful_voice_detection(audio_data, 16000)?;
```

## WAV File Support

The run_voice_detection_wav() function accepts WAV files as input. The function checks for valid RIFF headers and compatible sample rates (8000 Hz or 16000 Hz) before processing.

Example:

```rust
let results = vad_session.run_voice_detection_wav("example.wav")?;
```

## Resetting VAD State

To reset the session's internal state between different voice detection tasks, you can use the following methods:

>full_reset() : Resets both the hidden vectors (h, c) and the internal time counter.

>audio_state_reset(): Resets only the hidden vectors.

>time_reset(): Resets only the time counter.

Example:

```rust
vad_session.full_reset();
```

## Handling Unsupported Sample Rates

The model supports two sample rates: 8000 Hz and 16000 Hz. If an unsupported sample rate is detected an error is returned. Only when using WAV files can it be guarenteed to not run with an incompatible sample rate as the sample rate is directly pulled from the file data. The methods run_voice_detection() and run_stateful_voice_detection() could be run on data that is of the wrong sample rate if a sample rate of 8000 or 16000 is given as an argument. This will return incorrect results.

## Results

Results are returned as a struct: VoiceDetectResult.

This has 6 pieces of data:

>start_period: i32

Represents the start index of the audio segment where voice detection was analyzed. This is the starting point in the sample where the model began detecting speech.

>end_period: i32

Marks the end index of the audio segment being analyzed. It indicates the position in the sample where the detection window ends.

>probability: f32

A floating-point value that represents the probability that speech was detected within the analyzed segment. This value is between 0.0 and 1.0, where higher values indicate a stronger likelihood of speech presence.

>voice_detect: bool

A boolean flag indicating whether speech was detected or not. If true, speech is detected within the window, based on the probability threshold defined in the VadSession.

>relative_start_period: f64

Indicates the time in milliseconds when the detection window starts, relative to the total duration of the audio input processed so far.

>relative_end_period: f64

Similar to relative_start_period, but marks when the detection window ends in terms of total elapsed time, providing the relative position in the full audio file.

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

|Inputs:|
|---|
|0 input: Tensor<f32>(dyn, dyn)|
|1 sr: Tensor<i64>()|
|2 h: Tensor<f32>(2, dyn, 64)|
|3 c: Tensor<f32>(2, dyn, 64)|

|Outputs:|
|---|
|0 output: Tensor<f32>(dyn, 1)|
|1 hn: Tensor<f32>(2, dyn, 64)|
|2 cn: Tensor<f32>(2, dyn, 64)|


# Code Coverage

## LLVM on MacOS
```bash
export RUSTFLAGS="-Cinstrument-coverage"
cargo build
export LLVM_PROFILE_FILE="silero-%p-%m.profraw"
cargo test
grcov . -s . --binary-path ./target/debug/ -t html --branch --ignore-not-existing -o ./target/debug/coverage/
open ./target/debug/coverage/index.html
```

## Current Code Coverage Report

|Filename                     | Regions  |  Missed Regions  |   Cover  | Functions | Missed Functions | Executed   |    Lines   |   Missed Lines  |   Cover   | Branches  | Missed Branches   |
|-----------------------------|----------|------------------|----------|-----------|------------------|------------|------------|-----------------|-----------|-----------|-------------------|
|vad_environment.rs           |      24  |              14  |  41.67%  |         6 |                4 |   33.33%   |       40   |             17  |  57.50%   |        0  |               0   |
|vad_session.rs               |     178  |              39  |  78.09%  |        24 |                1 |   95.83%   |      259   |             17  |  93.44%   |        0  |               0   |
|TOTAL                        |     202  |              53  |  73.76%  |        30 |                5 |   83.33%   |      299   |             34  |  88.63%   |        0  |               0   |
