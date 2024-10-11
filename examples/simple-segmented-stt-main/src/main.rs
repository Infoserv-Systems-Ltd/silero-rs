use std::{
    io::{Read, Write},
    str::FromStr,
};

use std::path::{Path, PathBuf};

use clap::{arg, command, Parser};
use serde::{Deserialize, Serialize};

use silero_rs::vad_session::VadSession;

pub mod stt_segment;
pub mod whisper_wrapper;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentedSttConfig {
    model_path: String,
    audio_chunk_size_secs: usize,
    threads: u8,
    greedy_best_of: u8,
}

#[derive(Debug, Serialize, Deserialize)]
struct TracingConfig {
    level: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    segmented_stt: SegmentedSttConfig,
    tracing: TracingConfig,
}

/// Config file for the segmented STT tool
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path/location of the config file
    #[arg(long)]
    config: String,
    #[arg(long)]
    file: String,
    #[arg(long)]
    output: Option<String>,
}

fn main() {
    let args = Args::parse();

    let config_file = std::fs::File::open(args.config).expect("Could not open config file");
    let config = serde_yaml::from_reader::<_, Config>(config_file)
        .expect("Could not read values from config file");

    let trace_level = tracing::Level::from_str(&config.tracing.level)
        .expect("unable to determine trace level in config");

    // a builder for `FmtSubscriber`.
    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        // all spans/events with a level higher than TRACE (e.g, debug, info, warn, etc.)
        // will be written to stdout.
        .with_max_level(trace_level)
        // completes the builder.
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    tracing::info!("starting speech to test segmenter");

    let whisper = whisper_wrapper::WhisperWrapper::new(
        config.segmented_stt.threads,
        config.segmented_stt.greedy_best_of,
        &config.segmented_stt.model_path,
    );

    let mut audio_file = std::fs::File::open(args.file).expect("Could not open audio file");

    let mut whisper_state = whisper.ctx.create_state().unwrap();

    let mut buffer = vec![
        0;
        whisper_wrapper::AUDIO_SAMPLE_RATE
            * (whisper_wrapper::AUDIO_SAMPLE_SIZE / 8)
            * config.segmented_stt.audio_chunk_size_secs
    ];

    let output_file = args.output.map(|output_filename| {
        std::fs::File::create(output_filename).expect("could not create output file")
    });

    let mut silero_session: VadSession = VadSession::new(&PathBuf::from("models/silero_vad.onnx"), 0.5).unwrap();

    // this is the externally managed timing reference for teh audio
    // for some reason, whisper doesn't keep state of the audio time
    // even when you hold a reference to its "state". All teh timings it returns
    // are offsets related to that particular audio segment. So we keep track of
    // where we are so when segment 3 states audio was at 3 seconds, we can
    // determine it is 6 + 6 + 3 seconds into the audio file
    // this is why we need to have timings from teh VAD (Voice Activity Detection)
    // so we can hold reference to teh timings in the context of the complete audio file
    let mut wallclock_timer = 0;
    loop {
        let bytes_read = audio_file.read(&mut buffer[..]);

        if bytes_read.is_err() {
            tracing::error!("error reading from file");
            return;
        }

        let bytes_read = bytes_read.unwrap();
        tracing::debug!("read {} bytes", bytes_read);
        if bytes_read == 0_usize {
            break;
        }

        let buffer = unsafe { buffer.align_to::<i16>().1 };

        let now = std::time::Instant::now();

        let params = whisper.get_full_params();


        let fragments = whisper_wrapper::push_audio_i16(
            params,
            &mut whisper_state,
            100 * wallclock_timer,
            whisper_wrapper::AUDIO_SAMPLE_RATE as u32,
            &buffer[..(bytes_read / (whisper_wrapper::AUDIO_SAMPLE_SIZE / 8))],
            &mut silero_session,
            &whisper
        );

        tracing::info!(
            "Vector being returned with: {} elements, {} seconds of audio, processed in {}mS",
            fragments.len(),
            config.segmented_stt.audio_chunk_size_secs,
            now.elapsed().as_millis()
        );

        for fragment in fragments {
            tracing::debug!("{:?}", fragment);

            tracing::info!(
                "Segment: {} - {} - {}",
                fragment.start_as_timestamp(),
                fragment.end_as_timestamp(),
                fragment.text()
            );

            if let Some(file) = output_file.as_ref().as_mut() {
                file.write_all(fragment.text().as_bytes())
                    .expect("error writing to output file");
            }
        }

        wallclock_timer += config.segmented_stt.audio_chunk_size_secs as i64;
    }

    tracing::info!(
        "processed, {} bytes (approx. {} secs of audio)",
        audio_file.metadata().unwrap().len(),
        audio_file.metadata().unwrap().len() as usize
            / ((whisper_wrapper::AUDIO_SAMPLE_SIZE / 8) * whisper_wrapper::AUDIO_SAMPLE_RATE),
    );
}
