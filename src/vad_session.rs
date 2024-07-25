#![allow(dead_code)]

use ort::{inputs, GraphOptimizationLevel, Session, Tensor};
use anyhow::{Error, Ok};
use ndarray::prelude::*;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::Read;
use hound::WavReader;

pub struct VADResult {
    start_period: i32,
    end_period: i32,
    probability: f32,
    voice_detect: bool
}

impl VADResult {
    pub fn new(start_period: i32, end_period: i32, probability: f32, voice_detect: bool) -> Self {
        Self {
            start_period,
            end_period,
            probability,
            voice_detect
        }
    }
}
pub struct AudioWindow {
    audio_data: Vec<f32>,
    start_period: i32,
    end_period: i32
}

impl AudioWindow {
    pub fn new(audio_data: Vec<f32>, start_period: i32, end_period: i32) -> Self {
        Self {
            audio_data,
            start_period,
            end_period
        }
    }
}
pub struct VadSession {
    h: ArrayBase<ndarray::OwnedRepr<f32>, ndarray::prelude::Dim<[usize; 3]>>,
    c: ArrayBase<ndarray::OwnedRepr<f32>, ndarray::prelude::Dim<[usize; 3]>>,
    vad_threshold: f32,
    session: ort::Session,
}

impl VadSession {
    pub fn new(model_location: &PathBuf, vad_threshold: f32) -> Result<Self, anyhow::Error> {
        //concerned with this due to lack of execution provider and environment which I used in silero_example however in ort documentation it says
        //that when an environment is not used one is made by default. It appears to be working on my end interested to know what happens on your end.
        //will what I have done here work for multiple streams I think there may be issues? Could make another object containing an ort environment that creates SileroVadOnnxModel instances
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_intra_threads(1)?
            .with_model_from_file(&model_location)?;

        Ok(Self {
            //assumes sample rate of data given is 16kHz could assign this later on to give flexability
            //same with vad_threshold
            //does the model location need to be saved?
            //do i need h and c here if each clip is independant? If this is the case the h/c need only exist in run_vad scope
            h: Array3::<f32>::zeros((2, 1, 64)),
            c: Array3::<f32>::zeros((2, 1, 64)),
            vad_threshold,
            session,
        })
    }

    pub fn run_vad(&mut self, audio_data: Vec<f32>, sample_rate: i64) -> Result<Vec<VADResult>, anyhow::Error> {
        
        if !((sample_rate == 8000) || (sample_rate == 16000)) {
            return Err(Error::msg("Sample rate must be 8000 or 16000"))
        }
        let audio_windows: Vec<AudioWindow> = prepare_data(audio_data, sample_rate);
        let mut audio_windows_result: Vec<VADResult> = Vec::new();

        let sample_rate = Array1::from(vec![sample_rate]);
        let mut h = Array3::<f32>::zeros((2, 1, 64));
        let mut c = Array3::<f32>::zeros((2, 1, 64));
        let h_c_dims = h.raw_dim();

        for window in audio_windows {

            let input_audio = ndarray::Array1::from_iter(window.audio_data.iter().cloned());
            let input_audio = input_audio.view().insert_axis(Axis(0));
            let window_outputs: &ort::SessionOutputs<'_> = &self.session.run(inputs![
                input_audio,
                sample_rate.clone(),
                h.clone(),
                c.clone()
            ]?)?;

            let window_output: Tensor<f32> = window_outputs["output"].extract_tensor()?;
            let window_output: Vec<f32> = window_output.view().iter().copied().collect();

            let hn: Tensor<f32> = window_outputs["hn"].extract_tensor()?;
            let hn: Vec<f32> = hn.view().iter().copied().collect();

            let cn: Tensor<f32> = window_outputs["cn"].extract_tensor()?;
            let cn: Vec<f32> = cn.view().iter().copied().collect();

            h = Array3::from_shape_vec((h_c_dims[0], h_c_dims[1], h_c_dims[2]), hn).unwrap();
            c = Array3::from_shape_vec((h_c_dims[0], h_c_dims[1], h_c_dims[2]), cn).unwrap();
            
            let probability: f32 = window_output.get(0).unwrap().clone();
            let voice_detect: bool;

            if probability < self.vad_threshold {
               voice_detect = false;
            } else {
                voice_detect = true;
            }

            let vad_result: VADResult = VADResult::new(window.start_period, window.end_period, probability, voice_detect);
            audio_windows_result.push(vad_result);
        }

        return Ok(audio_windows_result)
    }

    pub fn run_vad_wav(&mut self, file_path: &str) -> Result<Vec<VADResult>, anyhow::Error> {

        let path = Path::new(&file_path);
        if !path.exists() {
            return Err(Error::msg("File can connot be found at the supplied path"))
        }
        let mut file = File::open(&file_path)?;
        let mut header = [0; 4];
        file.read_exact(&mut header)?;
        if !(header == *b"RIFF") {
            return Err(Error::msg("Please use a WAV file"))
        }

        let mut reader = WavReader::open(file_path)?;
        let sample_rate: i64 = reader.spec().sample_rate as i64;
        let aud_array: Vec<i32> = reader.samples::<i32>().map(|x| x.unwrap()).collect::<Vec<_>>();
        let aud_array: Vec<f32> = aud_array.iter().map(|&x| x as f32).collect();

        if !(sample_rate == 8000 || sample_rate == 16000) {
            return Err(Error::msg("The sample rate of this audio file is not compatible with Silero, it must be 8000 or 16000 kHz"))
        }
        
        return self.run_vad(aud_array, sample_rate)
        
    }

    pub fn reset(&mut self) {
        self.h = Array3::<f32>::zeros((2, 1, 64));
        self.c = Array3::<f32>::zeros((2, 1, 64));
    }

    pub fn stateful_vad(&mut self, audio_data: Vec<f32>, sample_rate: i64) -> Result<Vec<VADResult>, anyhow::Error> {

        if !((sample_rate == 8000) || (sample_rate == 16000)) {
            return Err(Error::msg("Sample rate must be 8000 or 16000"))
        }
        let audio_windows: Vec<AudioWindow> = prepare_data(audio_data, sample_rate);
        let mut audio_windows_result: Vec<VADResult> = Vec::new();

        let sample_rate = Array1::from(vec![sample_rate]);
        let h_c_dims = self.h.raw_dim();

        for window in audio_windows {

            let input_audio = ndarray::Array1::from_iter(window.audio_data.iter().cloned());
            let input_audio = input_audio.view().insert_axis(Axis(0));
            let window_outputs: &ort::SessionOutputs<'_> = &self.session.run(inputs![
                input_audio,
                sample_rate.clone(),
                self.h.clone(),
                self.c.clone() 
            ]?)?;

            let window_output: Tensor<f32> = window_outputs["output"].extract_tensor()?;
            let window_output: Vec<f32> = window_output.view().iter().copied().collect();

            let hn: Tensor<f32> = window_outputs["hn"].extract_tensor()?;
            let hn: Vec<f32> = hn.view().iter().copied().collect();

            let cn: Tensor<f32> = window_outputs["cn"].extract_tensor()?;
            let cn: Vec<f32> = cn.view().iter().copied().collect();

            self.h = Array3::from_shape_vec((h_c_dims[0], h_c_dims[1], h_c_dims[2]), hn).unwrap();
            self.c = Array3::from_shape_vec((h_c_dims[0], h_c_dims[1], h_c_dims[2]), cn).unwrap();
            
            let probability: f32 = window_output.get(0).unwrap().clone();
            let voice_detect: bool;

            if probability < self.vad_threshold {
               voice_detect = false;
            } else {
                voice_detect = true;
            }

            let vad_result: VADResult = VADResult::new(window.start_period, window.end_period, probability, voice_detect);
            audio_windows_result.push(vad_result);
        }
        return Ok(audio_windows_result)
    }

    pub fn stateful_vad_wav(&mut self, file_path: &str) -> Result<Vec<VADResult>, anyhow::Error> {
        
        let path = Path::new(&file_path);
        if !path.exists() {
            return Err(Error::msg("File can connot be found at the supplied path"))
        }
        let mut file = File::open(&file_path)?;
        let mut header = [0; 4];
        file.read_exact(&mut header)?;
        if !(header == *b"RIFF") {
            return Err(Error::msg("Please use a WAV file"))
        }

        let mut reader = WavReader::open(file_path)?;
        let sample_rate: i64 = reader.spec().sample_rate as i64;
        let aud_array: Vec<i32> = reader.samples::<i32>().map(|x| x.unwrap()).collect::<Vec<_>>();
        let aud_array: Vec<f32> = aud_array.iter().map(|&x| x as f32).collect();

        if !(sample_rate == 8000 || sample_rate == 16000) {
            return Err(Error::msg("The sample rate of this audio file is not compatible with Silero, it must be 8000 or 16000 kHz"))
        }
        return self.stateful_vad(aud_array, sample_rate)
    }
}


fn prepare_data(audio_data: Vec<f32>, sample_rate: i64) -> Vec<AudioWindow> {
    let mut audio_windows: Vec<AudioWindow> = Vec::new();
    let no_of_windows = audio_data.len() / sample_rate as usize;

    if no_of_windows == 0 {
        let audio_window: AudioWindow = AudioWindow::new(audio_data.clone(), 0, audio_data.len().try_into().unwrap());
        audio_windows.push(audio_window);
        return audio_windows;
    }

    let mut start_index: usize = 0;
    for _ in 0..no_of_windows {
        let end_index: usize = start_index + sample_rate as usize;
        let window = &audio_data[start_index..end_index];
        let audio_window: AudioWindow = AudioWindow::new(window.to_vec(), start_index as i32, (end_index as i32)-1);
        audio_windows.push(audio_window);
        start_index = end_index;
    }

    if (audio_data.len() % sample_rate as usize) != 0 {
        let window = &audio_data[start_index..(audio_data.len() as usize)];
        let audio_window: AudioWindow = AudioWindow::new(window.to_vec(), start_index as i32, (audio_data.len()-1).try_into().unwrap());
        audio_windows.push(audio_window);
    }
    return audio_windows;
}

#[cfg(test)]
mod tests {

    use super::*;
    use hound::WavReader;
    use rand::prelude::*;

    fn get_audio_wav(file_path: &str) -> Result<Vec<f32>, anyhow::Error> {
        let mut reader = WavReader::open(file_path)?;
        let aud_array: Vec<i32> = reader
            .samples::<i32>()
            .map(|x| x.unwrap())
            .collect::<Vec<_>>();
        let aud_array: Vec<f32> = aud_array.iter().map(|&x| x as f32).collect();
        Ok(aud_array)
    }

    #[test]
    fn test_prepre_data() {
        let mut rng = rand::thread_rng();
        let random_vector: Vec<f32> = (0..100000).map(|_| rng.gen::<f32>()).collect();
        let result1 = prepare_data(random_vector, 8000);
        let test_sample = result1.get(0).unwrap().audio_data.clone();
        let test_sample2 = result1.get(12).unwrap().audio_data.clone();
        assert_eq!(result1.len(), 13);
        assert_eq!(test_sample.len(), 8000);
        assert_eq!(test_sample2.len(), 4000);

        let random_vector2: Vec<f32> = (0..100).map(|_| rng.gen::<f32>()).collect();
        let result2 = prepare_data(random_vector2, 16000);
        let test_sample3 = result2.get(0).unwrap().audio_data.clone();
        assert_eq!(test_sample3.len(), 100, "Prepare data is not working correctly");
    }

    #[test]
    fn test_struct() {
        let audio_data = get_audio_wav("files/example.wav").unwrap();
        let audio_data2 = audio_data.clone();
        let mut silero_model = VadSession::new(&PathBuf::from("files/silero_vad.onnx"), 0.5,).unwrap();
        let result = silero_model.run_vad(audio_data, 16000).unwrap();
        let result2 = prepare_data(audio_data2, 16000);
        assert_eq!(result.len(), result2.len());

        let expected_outputs: [bool; 35] = [
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true, true, true, true, true,
        ];

        for i in 0..result.len() {
            assert_eq!(result.get(i).unwrap().voice_detect, *expected_outputs.get(i).unwrap(), "run_vad_wav is not outputting the expected values");
        }
    }

    #[test]
    fn test_run_vad_wav () {
        let mut silero_model = VadSession::new(&PathBuf::from("files/silero_vad.onnx"), 0.5).unwrap();
        let result = silero_model.run_vad_wav("files/example.wav");
        match result {
            Ok(result) => {
            
                let expected_outputs: [bool; 35] = [
                    true, true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true,
                ];

                for i in 0..result.len() {
                    assert_eq!(result.get(i).unwrap().voice_detect, *expected_outputs.get(i).unwrap(), "run_vad is not outputting the expected values");
                }

            }

            Err(_) => {
                assert_eq!(true, false, "run_vad_wav not working");
            }
        }

    }
/*
need tests for:
stateful
new structs
 */
    #[test]
    fn test_stateful_interface () {
        let mut silero_model = VadSession::new(&PathBuf::from("files/silero_vad.onnx"), 0.5).unwrap();
        let result = silero_model.stateful_vad_wav("files/example.wav").unwrap();
        let expected_outputs: [bool; 35] = [
                    true, true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true,
                ];
        for i in 0..result.len() {
                assert_eq!(result.get(i).unwrap().voice_detect, *expected_outputs.get(i).unwrap(), "run_vad is not outputting the expected values");
        }

    }

}


