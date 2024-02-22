#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(dead_code)]

use anyhow::Error;
use clap::Parser;

use ndarray::{arr1, arr2, arr3, Array2, Array3};
use ndarray::{array, concatenate, s, Array1, Axis, CowArray, IxDyn, FixedInitializer};

use rand::seq::index::sample;
use rodio::{decoder::DecoderError, Decoder, Sample, Source};
use id3::Tag;
use hound::WavReader;

use ort::inputs;
use ort::GraphOptimizationLevel;
use ort::TensorElementDataType;
use ort::Value;
use ort::{CoreMLExecutionProvider, Session, Tensor};

use rand::Rng;

use std::array;
use std::convert::TryInto;
use std::error::Error as Err;
use std::fs::File;
use std::io::BufReader;
use std::any::type_name;
use std::process::Output;

use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use plotters::prelude::*;


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

pub struct AudioInfo {

    sample_rate: i64,
    audio_array: Vec<f32>,
}


fn main() -> Result<(), anyhow::Error>{

    let args = Args::parse(); 

    let model_result = run_model(&args.model, &args.audio)?;
  
    Ok(())
}

fn get_audio_mp3(file_path: &str) -> Result<Vec<f32>, anyhow::Error> {

    let aud_file = File::open(file_path)?;
    let aud_decoder = Decoder::new(BufReader::new(aud_file))?;
    let aud_array: Vec<i16> = aud_decoder.collect();
    let aud_array: Vec<f32> = aud_array.iter().map(|&x| x as f32).collect();

    /*match Tag::read_from_path(file_path) {
        Ok(tag) => {
            if let Some(sr) = tag.sample_rate() {
                sample_rate = sr;
            } else {
                return Error(println!("File has no metadata"));
            }
        }
        Err(err) => {
            eprintln!("Error reading MP3 file metadata: {}", err);
        }
    }*/



    Ok(aud_array)
}

fn get_audio_wav(file_path: &str) -> Result<AudioInfo, anyhow::Error> {

    let mut reader = WavReader::open(file_path)?;
    let samp_rate: i64 = reader.spec().sample_rate as i64;
    let aud_array: Vec<i32> = reader.samples::<i32>().map(|x| x.unwrap()).collect::<Vec<_>>();
    let aud_array: Vec<f32> = aud_array.iter().map(|&x| x as f32).collect();
    println!("{}", samp_rate);
    let return_aud: AudioInfo = AudioInfo {
        sample_rate: samp_rate,
        audio_array: aud_array,
    };

    Ok(return_aud)
}
fn run_model(model_location: &str, audio_location: &str) -> ort::Result<()> {

    let audio_data = get_audio_wav(audio_location).unwrap();
    let audio_vec = audio_data.audio_array;
    let mut no_of_windows = (audio_vec.len() / 16000) as i16;
    let len_final_window = (audio_vec.len() % 16000) as i16;
    let mut audio_windows: Vec<Vec<f32>> = Vec::new();
    //could put this in a method?
    if no_of_windows == 0 {
        audio_windows.push(audio_vec.clone());

    } else {

        //Works up to 74 hours
        let mut current_start: usize = 0;
        for i in 0..no_of_windows {
            let mut temp_vec: Vec<f32> = Vec::new();
            let end_index: usize = current_start + 16000;
            let window = &audio_vec[current_start..end_index];
            temp_vec.extend_from_slice(window);
            audio_windows.push(temp_vec);
            current_start = end_index;
        }

        if len_final_window != 0 {
            let mut temp_vec: Vec<f32> = Vec::new();
            let end_index: usize = audio_vec.len() as usize;
            let window = &audio_vec[current_start..end_index];
            temp_vec.extend_from_slice(window);
            audio_windows.push(temp_vec);
        }
    }

    ort::init()
        .with_name("Silero-VAD")
        .with_execution_providers([CoreMLExecutionProvider::default().build()])
        .commit()?;

    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .with_model_from_file(model_location);


    let mut input_audio_data = Array1::from_iter(audio_vec.iter().cloned());
    let input_audio_data = input_audio_data.view().insert_axis(Axis(0));

    let sr: i64 = audio_data.sample_rate;
    println!("{}", sr);
    let sr_array = Array1::from(vec![sr]);
    
    let mut h = make_h_c_array();
    let mut c = make_h_c_array();
    let dims = h.raw_dim();
    
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    let session_binding = session?;
    //let mut window_avg: Vec<f32> = Vec::new();

    for i in 0..audio_windows.len() {

        let mut current_window: Vec<f32> = audio_windows.get(i).unwrap().clone();
        let mut current_window = Array1::from_iter(current_window.iter().cloned());
        let current_window = current_window.view().insert_axis(Axis(0));
        let temp_sr = sr_array.clone();
        let temp_h = h.clone();
        let temp_c = c.clone();

        let output = session_binding.run(inputs![current_window, temp_sr, temp_h, temp_c]?)?;

        let window_output: Tensor<f32> = output["output"].extract_tensor()?;
        let hn: Tensor<f32> = output["hn"].extract_tensor()?;
        let cn: Tensor<f32> = output["cn"].extract_tensor()?;

        let window_output: ort::ArrayViewHolder<'_, f32> = window_output.view();
        let window_output: Vec<f32> = window_output.iter().copied().collect();
        outputs.push(window_output);

        let hn = hn.view();
        let cn = cn.view();

        let hn: Vec<f32> = hn.iter().copied().collect();
        let cn: Vec<f32> = cn.iter().copied().collect();

        h = Array3::from_shape_vec((dims[0], dims[1], dims[2]), hn).unwrap();
        c = Array3::from_shape_vec((dims[0], dims[1], dims[2]), cn).unwrap();

    }
    
    let outputs: Vec<f32> = outputs.into_iter().flat_map(|f| f ).collect();
    println!("{:?}", outputs);

    Ok(())
    
}

fn make_h_c_array() -> Array3<f32> {

    let a = Array3::<f32>::zeros((2, 1, 64));
    return a;
}

/*fn create_graphs(input_audio: &Vec<f32>, output: &Vec<f32>) -> Result<(), anyhow::Error> {

    let mut max: f32 = 0.0;
    let mut graph_input_vec: Vec<f32> = Vec::new();
    let mut graph_output_vec: Vec<f32> = Vec::new();

    for i in input_audio {
        if i > &max {
            max = *i;
        }
    }

    for i in 0..input_audio.len() {

        let mut temp: f32 = *graph_input_vec.get(i).unwrap();
        temp = temp/&max;
        graph_input_vec.push(temp);
    }

    for i in output {
        for j in 0..1536 {
            graph_output_vec.push(i.clone());
        }
    }

    if graph_input_vec.len() > graph_output_vec.len() {

        graph_output_vec.truncate(graph_input_vec.len());
    }

    let root = BitMapBackend::new("waveform.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_value = 1;
    let min_value = 0;

    let mut chart = ChartBuilder::on(&root)
    .x_label_area_size(30)
    .y_label_area_size(30)
    .margin(5)
    .caption("Waveform", ("Arial", 30).into_font())
    .build_cartesian_2d(0..graph_input_vec.len() as i32, 0..1)?;

    chart.draw_series(LineSeries::new(
        (0..input_audio.len()).map(|i| (i as i32, samples[i])),
        &RED,
    ))?;


    Ok(())
}*/