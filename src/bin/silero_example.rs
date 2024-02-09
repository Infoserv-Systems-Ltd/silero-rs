#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]

use clap::Parser;

use ndarray::{array, concatenate, s, Array1, Axis, CowArray, IxDyn};

use rodio::{decoder::DecoderError, Decoder, Sample, Source};

use ort::inputs;
use ort::GraphOptimizationLevel;
use ort::TensorElementDataType;
use ort::Value;
use ort::{CoreMLExecutionProvider, Session, Tensor};

use rand::Rng;

use std::convert::TryInto;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use tracing::Level;
use tracing_subscriber::FmtSubscriber;


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path/location of the model
    #[arg(long)]
    pub model: String,

    /// Path/location of the audio file
    #[arg(long)]
    pub audio: String,
}

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

fn main() -> Result<(), anyhow::Error>{

    let args = Args::parse();

    let silero_info =
        get_model_info(&args.model)?;
    

    let audio_sample_result =
        get_audio(&args.audio)?;

    let model_result = run_model(&args.model, &audio_sample_result)?;
  
    Ok(())
}

fn get_audio(file_path: &str) -> Result<Vec<f32>, anyhow::Error> {
    let aud_file = File::open(file_path)?;
    let aud_decoder = Decoder::new(BufReader::new(aud_file))?;
    let aud_array: Vec<i16> = aud_decoder.collect();
    
    // you can do away with this by tagging a map() conversion onto the above
    let mut return_arr = Vec::new();
    for item in aud_array {
        return_arr.push(item as f32);
    }

    Ok(return_arr)
}

fn display_element_type(t: TensorElementDataType) -> &'static str {
    match t {
        TensorElementDataType::Bfloat16 => "bf16",
        TensorElementDataType::Bool => "bool",
        TensorElementDataType::Float16 => "f16",
        TensorElementDataType::Float32 => "f32",
        TensorElementDataType::Float64 => "f64",
        TensorElementDataType::Int16 => "i16",
        TensorElementDataType::Int32 => "i32",
        TensorElementDataType::Int64 => "i64",
        TensorElementDataType::Int8 => "i8",
        TensorElementDataType::String => "str",
        TensorElementDataType::Uint16 => "u16",
        TensorElementDataType::Uint32 => "u32",
        TensorElementDataType::Uint64 => "u64",
        TensorElementDataType::Uint8 => "u8",
    }
}

// TODO: Fix this function
// the issue with teh latest (v2) ort crate is ort::value appears to be private
//fn display_value_type(value: &ort::value::ValueType) -> String {
//     match value {
//         ValueType::Tensor { ty, dimensions } => {
//             format!(
//                 "Tensor<{}>({})",
//                 display_element_type(*ty),
//                 dimensions
//                     .iter()
//                     .map(|c| if *c == -1 {
//                         "dyn".to_string()
//                     } else {
//                         c.to_string()
//                     })
//                     .collect::<Vec<_>>()
//                     .join(", ")
//             )
//         }
//         ValueType::Map { key, value } => format!(
//             "Map<{}, {}>",
//             display_element_type(*key),
//             display_element_type(*value)
//         ),
//         ValueType::Sequence(inner) => format!("Sequence<{}>", display_value_type(inner)),
//     }
// }

fn get_model_info(model_path: &str) -> Result<(), anyhow::Error> {
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

        // TODO: Fix this
        // model_info.push_str(&format!(" {}\n", display_value_type(&input.input_type)));
    }
    model_info.push_str("Outputs:\n");
    for (i, output) in session.outputs.iter().enumerate() {
        model_info.push_str(&format!("   {}", i.to_string()));
        model_info.push_str(&format!(" {}:", output.name));
        // TODO: Fix this
        // model_info.push_str(&format!(" {}\n", display_value_type(&output.output_type)));
    }
    println!("{}", model_info.to_string());

    Ok(())
}

fn run_model(model_location: &str, data_sample: &Vec<f32>) -> ort::Result<()> {
    ort::init()
        .with_name("Silero-VAD")
        .with_execution_providers([CoreMLExecutionProvider::default().build()])
        .commit()?;

    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .with_model_from_file(model_location);

    let mut input_array = Array1::from_iter(data_sample.iter().cloned());
    let input_array2 = input_array.view().insert_axis(Axis(0));

    let sr: i64 = 8000;
    let sr_array = Array1::from(vec![sr]);

    let arr_1: Array1<f32> = array![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ];
    let arr_1 = arr_1.insert_axis(ndarray::Axis(0));
    //let cow_arr: CowArray<'_, f32, IxDyn> = CowArray::from(arr_1);

    let arr_2 = array![arr_1.clone(), arr_1.clone()];

    let h = arr_2.clone();
    let c = arr_2.clone();

    //let c_array = Array1::from(vec![c]);

    let binding = session?;
    //not progressing past this line
    // TODO: uncomment the lines below
    // let outputs = binding.run(inputs![input_array2, sr_array, h, c]?)?;
    // let generated_tokens: Tensor<f32> = outputs["output1"].extract_tensor()?;
    // let generated_tokens = generated_tokens.view();
    // println!("{}", generated_tokens.get(0).unwrap());

    Ok(())
}

fn create_rand_array(length: i32) -> Vec<(f32, f32)> {
    let mut arr: Vec<(f32, f32)> = Vec::new();
    let mut rng = rand::thread_rng();

    for i in 0..length {
        let int1: f32 = rng.gen_range(0..1000) as f32;
        let int2: f32 = rng.gen_range(0..10000) as f32;
        arr.push((int1, int2));
    }
    return arr;
}
