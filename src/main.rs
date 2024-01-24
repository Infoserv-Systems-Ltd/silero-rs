#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
use onnxruntime::{environment::Environment, LoggingLevel, GraphOptimizationLevel, OrtError, session::{self, Input}};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;
use rodio::{Decoder, Sample, Source, decoder::DecoderError};
use std::fs::File;
use std::io::BufReader;
use std::error::Error;


fn main() {
    
    let audio_sample_result = get_audio("/Users/oliver.pikett/Documents/SileroDev/my_work/files/10-seconds.mp3");

    match audio_sample_result {

        Ok(audio_sample) => {
            
            println!("Running langauge model");
            let program_result = run(audio_sample);
            match program_result {

                Ok(program_ran) => {
                    println!("Program ran sucessfully");
                }

                Err(err) => {
                    println!("~~~Language model encounted an error:");
                    println!("{}~~~", err);
                    std::process::exit(1);
                }
            }
        }

        Err(err) => {
            std::process::exit(1);
        }
    }
   
}

fn get_audio(file_path: &str) -> Result<Vec<i16>, Box<dyn Error>> {

    let aud_file: File = File::open(file_path)?;
    let aud_decoder: Decoder<BufReader<File>> = Decoder::new(BufReader::new(aud_file))?;
    let aud_array: Vec<i16> = aud_decoder.collect();
    Ok(aud_array)
}
 
fn run(data_sample: Vec<i16>) -> Result<(), Box<dyn Error>> {

    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    println!("~~~Initalising Environment~~~");
    let environment = Environment::builder()
        .with_name("test")
        // The ONNX Runtime's log level can be different than the one of the wrapper crate or the application.
        .with_log_level(LoggingLevel::Info)
        .build()?;

    println!("~~~Initalising Session~~~");
    let mut session = environment

        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::Basic)?
        .with_number_threads(1)?
        .with_model_from_file("/Users/oliver.pikett/Documents/SileroDev/my_work/files/silero_vad.onnx")?;
        //.with_model_from_file("/Users/oliver.pikett/Documents/SileroDev/my_work/files/alexnet_Opset18.onnx")?;

    println!("~~~Session successfully initialised~~~");

    let input_info = &session.inputs;
    for input in input_info {
        println!("Input Name: {}", input.name);
        println!("Input Type: {:?}", input.dimensions);
    }

    Ok(())

}
