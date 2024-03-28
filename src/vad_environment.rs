use crate::vad_session::VadSession;
use anyhow::Error;
use ort::Session;
use std::path::{Path, PathBuf};

//anyway to get these things to not come up as unsued without doing allow deadcode at start of file?
pub struct VadEnvironment {
    model_location: PathBuf,
    //keep vad_sessions in vector?
    vad_sessions: Vec<Session>,
}

impl VadEnvironment{
    pub fn new(model_location: &str) -> Result<Self, anyhow::Error> {
        //discuss whether to change sample rate and vad threshold on a per environment or
        //per session basis, will add in the ability to select vad threshold and
        //store sessions in vector and drop perodically?
        let path = Path::new(&model_location);
        if path.exists() {
            let model_location = PathBuf::from(model_location);
            let vad_sessions: Vec<Session> = Vec::new();

            Ok(Self {
                model_location,
                vad_sessions,
            })
        } else {
            Err(Error::msg("Model can connot be found at the supplied path"))
        }
    }

    pub fn new_vad_session(&self, sample_rate: i64) -> Result<VadSession, anyhow::Error> {
        let vad_session: Result<VadSession, anyhow::Error> = VadSession::new(&self.model_location, sample_rate);
        return vad_session;
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use hound::WavReader;

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
    fn test_struct() {
        let vad_environment = VadEnvironment::new("files/silero_vad.onnx").unwrap();
        let mut session = vad_environment.new_vad_session(16000).unwrap();
        let audio_data = get_audio_wav("files/example.wav").unwrap();
        let result = session.run_vad(audio_data, session.sample_rate).unwrap();

        let expected_outputs: [bool; 35] = [
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true, true, true, true, true, true, true, true, true, true, true, true,
            true, true, true, true, true, true, true,
        ];

        for i in 0..result.len() {
            assert_eq!(result.get(i).unwrap(), expected_outputs.get(i).unwrap());
        }
    }

    #[test]
    fn test_model_path_check() {

        let path = Path::new("files/silero_vad.onnx");
        assert_eq!(true, path.exists(), "Path not valid");
        let environment_error = VadEnvironment::new("not a path");
        match environment_error {
            Ok(_) => {
                assert_eq!(true, false)
            }
            Err(_) => {
                assert_eq!(true, true)
            }
        }

        let environment_ok = VadEnvironment::new("files/silero_vad.onnx");
        match environment_ok {
            Ok(_) => {
                assert_eq!(true, true)
            }
            Err(_) => {
                assert_eq!(true, false)
            }
        }
    }
}
