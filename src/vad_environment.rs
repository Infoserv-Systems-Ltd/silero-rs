
use ort::Session;

use anyhow::Error;

use crate::vad_session::VadSession;

//anyway to get these things to not come up as unsued without doing allow deadcode at start of file?
pub struct VadEnvironment {

    model_location: String,
    //keep vad_sessions in vector?
    vad_sessions: Vec<Session>,

}

impl VadEnvironment {

    pub fn new(model_location: String) -> Result<Self, Error> {

        //discuss whether to change sample rate and vad threshold on a per environment or
        //per session basis, will add in the ability to select vad threshold and 
        //store sessions in vector and drop perodically?
        let model_location = model_location;
        let vad_sessions: Vec<Session> = Vec::new();
        
        Ok(Self {

            model_location,
            vad_sessions
        })
    }
    
    pub fn new_vad_session(&self) -> Result<VadSession, Error> {

        let vad_session: Result<VadSession, Error> = VadSession::new(&self.model_location);
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
    fn test_struct () {

        let vad_environment = VadEnvironment::new(("/Users/oliver.pikett/Documents/GitHub/silero-rs/files/silero_vad.onnx").to_string()).unwrap();
        let mut session = vad_environment.new_vad_session().unwrap();
        let audio_data = get_audio_wav("/Users/oliver.pikett/Documents/GitHub/silero-rs/files/example.wav").unwrap();
        let result = session.run_vad(audio_data).unwrap();

        let expected_outputs: [bool; 35] = [true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
         true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true];

        for i in 0..result.len() {
            assert_eq!(result.get(i).unwrap(), expected_outputs.get(i).unwrap());
        }

    }
}