use crate::vad_session::VadSession;
use anyhow::Error;
use ort::Session;
use std::path::{Path, PathBuf};

//anyway to get these things to not come up as unsued without doing allow deadcode at start of file?
pub struct VadEnvironment {
    model_location: PathBuf,
    return_probs: bool,
    vad_threshold: f32,
    vad_sessions: Vec<Session>,
}

impl VadEnvironment{
    pub fn new(model_location: &str, return_probs: bool, vad_threshold: f32) -> Result<Self, anyhow::Error> {
        //discuss whether to change sample rate and vad threshold on a per environment or
        //per session basis, will add in the ability to select vad threshold and
        //store sessions in vector and drop perodically?
        let path = Path::new(&model_location);
        if path.exists() {
            let model_location = PathBuf::from(model_location);
            let vad_sessions: Vec<Session> = Vec::new();

            Ok(Self {
                model_location,
                return_probs,
                vad_threshold,
                vad_sessions,
            })
        } else {
            Err(Error::msg("Model can connot be found at the supplied path"))
        }
    }

    pub fn new_vad_session(&self) -> Result<VadSession, anyhow::Error> {
        let vad_session: Result<VadSession, anyhow::Error> = VadSession::new(&self.model_location, self.vad_threshold);
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
    fn test_model_path_check() {

        let path = Path::new("files/silero_vad.onnx");
        assert_eq!(true, path.exists(), "Path not valid");
        let environment_error = VadEnvironment::new("not a path", false, 0.5);
        match environment_error {
            Ok(_) => {
                assert_eq!(true, false)
            }
            Err(_) => {
                assert_eq!(true, true)
            }
        }

        let environment_ok = VadEnvironment::new("files/silero_vad.onnx", false, 0.5);
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
