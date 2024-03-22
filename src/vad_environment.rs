
use ort::Session;
use anyhow::Error;
use std::path::Path;
use crate::vad_session::VadSession;

//anyway to get these things to not come up as unsued without doing allow deadcode at start of file?
pub struct VadEnvironment {

    model_location: String,
    //keep vad_sessions in vector?
    vad_sessions: Vec<Session>,

}

impl VadEnvironment {

    pub fn new(model_location: String) -> Result<Self, anyhow::Error> {

        //discuss whether to change sample rate and vad threshold on a per environment or
        //per session basis, will add in the ability to select vad threshold and 
        //store sessions in vector and drop perodically?
        if Path::new(&model_location).exists() {
            let model_location = model_location;
            let vad_sessions: Vec<Session> = Vec::new();

            Ok(Self {
            model_location,
            vad_sessions
            })

        } else {
            Err(Error::msg("Model can connot be found at the supplied path"))
        }
    }
    
    pub fn new_vad_session(&self) -> Result<VadSession, anyhow::Error> {

        let vad_session: Result<VadSession, anyhow::Error> = VadSession::new(&self.model_location);
        return vad_session
        
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use hound::WavReader;

    fn get_audio_wav(file_path: &str) -> Result<Vec<f32>, anyhow::Error> {

    let mut reader = WavReader::open(file_path)?;
    let aud_array: Vec<i32> = reader.samples::<i32>().map(|x| x.unwrap()).collect::<Vec<_>>();
    let aud_array: Vec<f32> = aud_array.iter().map(|&x| x as f32).collect();
    Ok(aud_array)
    }

    #[test]
    fn test_struct() {

        let vad_environment = VadEnvironment::new(("files/silero_vad.onnx").to_string()).unwrap();
        let mut session = vad_environment.new_vad_session().unwrap();
        let audio_data = get_audio_wav("files/example.wav").unwrap();
        let result = session.run_vad(audio_data).unwrap();

        let expected_outputs: [bool; 35] = [true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
         true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true];

        for i in 0..result.len() {
            assert_eq!(result.get(i).unwrap(), expected_outputs.get(i).unwrap());
        }

    }

    #[test]
    fn test_model_path_check() {

        let invalid_path = "not_a_path".to_string();
        let environment_error = VadEnvironment::new(invalid_path);
        match environment_error {
            Ok(_) => {assert_eq!(true, false)}
            Err(_) => {assert_eq!(true, true)}
        }

        let valid_path = "files/silero_vad.onnx".to_string();
        let environment_ok = VadEnvironment::new(valid_path);
        match environment_ok {
            Ok(_) => {assert_eq!(true, true)}
            Err(_) => {assert_eq!(true, false)}
        }

    }
}