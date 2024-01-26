#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
//use onnxruntime::{environment::Environment, LoggingLevel, GraphOptimizationLevel, OrtError, session::{self, Input}};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;
use rodio::{Decoder, Sample, Source, decoder::DecoderError};
use std::fs::File;
use std::io::BufReader;
use std::error::Error;
use ort::{Session, TensorElementType, ValueType, GraphOptimizationLevel, Tensor, CoreMLExecutionProvider};

/*
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
*/

fn main() {
    
    let model_location: &str = "/Users/oliver.pikett/Documents/SileroDev/my_work/files/silero_vad.onnx";
    let model_info_result = get_model_info(&model_location);

    match model_info_result {
        
        Ok(model_info) => {
            println!("{}", model_info);
        }

        Err(err) => {
            println!("{}", err);
        }
    }

    let audio_sample_result = get_audio("/Users/oliver.pikett/Documents/SileroDev/my_work/files/10-seconds.mp3");
    match audio_sample_result {

        Ok(audio_sample) => {
            
            println!("Running langauge model");
            let program_result = run(&model_location, &audio_sample);
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

fn display_element_type(t: TensorElementType) -> &'static str {
	match t {
		TensorElementType::Bfloat16 => "bf16",
		TensorElementType::Bool => "bool",
		TensorElementType::Float16 => "f16",
		TensorElementType::Float32 => "f32",
		TensorElementType::Float64 => "f64",
		TensorElementType::Int16 => "i16",
		TensorElementType::Int32 => "i32",
		TensorElementType::Int64 => "i64",
		TensorElementType::Int8 => "i8",
		TensorElementType::String => "str",
		TensorElementType::Uint16 => "u16",
		TensorElementType::Uint32 => "u32",
		TensorElementType::Uint64 => "u64",
		TensorElementType::Uint8 => "u8"
	}
}

fn display_value_type(value: &ValueType) -> String {
	match value {
		ValueType::Tensor { ty, dimensions } => {
			format!(
				"Tensor<{}>({})",
				display_element_type(*ty),
				dimensions
					.iter()
					.map(|c| if *c == -1 { "dyn".to_string() } else { c.to_string() })
					.collect::<Vec<_>>()
					.join(", ")
			)
		}
		ValueType::Map { key, value } => format!("Map<{}, {}>", display_element_type(*key), display_element_type(*value)),
		ValueType::Sequence(inner) => format!("Sequence<{}>", display_value_type(inner))
	}
}

fn get_model_info(model_path: &str) -> Result<String, Box<dyn Error>>{

    let session = Session::builder()?.with_model_from_file(model_path)?;
    let mut model_info = String::new();
    let meta = session.metadata()?;
	if let Ok(x) = meta.name() {
        model_info.push_str(&format!("Name: {}\n", x));
	}
	if let Ok(x) = meta.description() {
		model_info.push_str(&format!("Description: {}\n", x));
	}
	if let Ok(x) = meta.producer() {
		model_info.push_str(&format!("Produced by {}\n", x));
	}

    model_info.push_str("Inputs:\n");
	for (i, input) in session.inputs.iter().enumerate() {
        model_info.push_str(&format!("   {}", i.to_string()));
        model_info.push_str(&format!(" {}:", input.name));
        model_info.push_str(&format!(" {}\n", display_value_type(&input.input_type)));
	}
	model_info.push_str("Outputs:\n");
	for (i, output) in session.outputs.iter().enumerate() {
        model_info.push_str(&format!("   {}", i.to_string()));
        model_info.push_str(&format!(" {}:", output.name));
        model_info.push_str(&format!(" {}\n", display_value_type(&output.output_type)));

	}

    Ok(model_info.to_string())

}

fn run(model_location: &str, data_sample: &Vec<i16>) -> Result<(), Box<dyn Error>> {

    ort::init()
        .with_name("Silero-VAD")
        .with_execution_providers([CoreMLExecutionProvider::default().build()])
        .commit()?;
    
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .with_model_from_file(model_location);


    Ok(())
}