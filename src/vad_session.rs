#![allow(dead_code)]

use ort::{inputs, GraphOptimizationLevel, Session, Tensor};

use ndarray::prelude::*;

pub struct VadSession {
    sample_rate:
        ArrayBase<ndarray::OwnedRepr<i64>, ndarray::prelude::Dim<[usize; 1]>>,
    h: ArrayBase<ndarray::OwnedRepr<f32>, ndarray::prelude::Dim<[usize; 3]>>,
    c: ArrayBase<ndarray::OwnedRepr<f32>, ndarray::prelude::Dim<[usize; 3]>>,
    vad_threshold: f32,
    session: ort::Session,
}

impl VadSession {
    pub fn new(model_location: &str) -> Result<Self, anyhow::Error> {
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
            sample_rate: Array1::from(vec![16000]),
            //do i need h and c here if each clip is independant? If this is the case the h/c need only exist in run_vad scope
            h: Array3::<f32>::zeros((2, 1, 64)),
            c: Array3::<f32>::zeros((2, 1, 64)),
            vad_threshold: 0.5,
            session,
        })
    }

    //will return true/false for 1sec chunks, final chunk could be too small to be worthwhile
    pub fn run_vad(&mut self, audio_data: Vec<f32>) -> Result<Vec<bool>, anyhow::Error> {
        //put in check to validate input here?
        let audio_chunks: Vec<Vec<f32>> = prepare_data(audio_data);
        let h_c_dims = self.h.raw_dim();
        let mut audio_chunks_result: Vec<bool> = Vec::new();

        for chunk in audio_chunks {
            let input_chunk = ndarray::Array1::from_iter(chunk.iter().cloned());
            let input_chunk = input_chunk.view().insert_axis(Axis(0));
            //is cloning necessary?
            let chunk_outputs: &ort::SessionOutputs<'_> = &self.session.run(inputs![
                input_chunk,
                self.sample_rate.clone(),
                self.h.clone(),
                self.c.clone()
            ]?)?;
            let chunk_output: Tensor<f32> = chunk_outputs["output"].extract_tensor()?;
            let chunk_output: Vec<f32> = chunk_output.view().iter().copied().collect();

            let hn: Tensor<f32> = chunk_outputs["hn"].extract_tensor()?;
            let hn: Vec<f32> = hn.view().iter().copied().collect();

            let cn: Tensor<f32> = chunk_outputs["cn"].extract_tensor()?;
            let cn: Vec<f32> = cn.view().iter().copied().collect();

            self.h = Array3::from_shape_vec((h_c_dims[0], h_c_dims[1], h_c_dims[2]), hn).unwrap();
            self.c = Array3::from_shape_vec((h_c_dims[0], h_c_dims[1], h_c_dims[2]), cn).unwrap();

            if chunk_output.get(0).unwrap() < &self.vad_threshold {
                audio_chunks_result.push(false);
            } else {
                audio_chunks_result.push(true);
            }
        }

        Ok(audio_chunks_result)
    }

    pub fn reset(&mut self) {
        self.h = Array3::<f32>::zeros((2, 1, 64));
        self.c = Array3::<f32>::zeros((2, 1, 64));
    }
}

//check audio_data is not empty BEFORE this method is called - too short/empty?
fn prepare_data(audio_data: Vec<f32>) -> Vec<Vec<f32>> {
    let mut audio_chunks: Vec<Vec<f32>> = Vec::new();
    let no_of_chunks = audio_data.len() / 16000;

    if no_of_chunks == 0 {
        audio_chunks.push(audio_data);
        return audio_chunks;
    }

    //more efficient way to do this?
    let mut start_index: usize = 0;
    for _ in 0..no_of_chunks {
        let end_index: usize = start_index + 16000;
        let chunk = &audio_data[start_index..end_index];
        audio_chunks.push(chunk.to_vec());
        start_index = end_index;
    }

    if (audio_data.len() % 16000) != 0 {
        let chunk = &audio_data[start_index..(audio_data.len() as usize)];
        audio_chunks.push(chunk.to_vec());
    }
    return audio_chunks;
}

//more tests needed? edge cases etc?
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
        let result1 = prepare_data(random_vector);
        let test_sample = result1.get(0).unwrap();
        let test_sample2 = result1.get(6).unwrap();
        assert_eq!(result1.len(), 7);
        assert_eq!(test_sample.len(), 16000);
        assert_eq!(test_sample2.len(), 4000);

        let random_vector2: Vec<f32> = (0..100).map(|_| rng.gen::<f32>()).collect();
        let result2 = prepare_data(random_vector2);
        let test_sample3 = result2.get(0).unwrap();
        assert_eq!(test_sample3.len(), 100);
    }

    #[test]
    fn test_struct() {
        let audio_data = get_audio_wav("files/example.wav").unwrap();
        let audio_data2 = audio_data.clone();
        let mut silero_model = VadSession::new("files/silero_vad.onnx").unwrap();
        let result = silero_model.run_vad(audio_data).unwrap();
        let result2 = prepare_data(audio_data2);
        assert_eq!(result.len(), result2.len());

        let expected_outputs: [bool; 35] = [
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true, true, true, true, true,
        ];

        for i in 0..result.len() {
            assert_eq!(result.get(i).unwrap(), expected_outputs.get(i).unwrap());
        }
    }
}

/*
to do:
nicer text for error messages
store file or path?
*/
