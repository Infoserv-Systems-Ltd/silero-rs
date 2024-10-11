use core::time;

use whisper_rs::FullParams;
use whisper_rs::SamplingStrategy;
use whisper_rs::WhisperContext;
use whisper_rs::WhisperContextParameters;
use whisper_rs::WhisperState;
use crate::stt_segment::STTSegment;

pub const AUDIO_SAMPLE_RATE: usize = 16_000; // Note: this is fixed for whisper and various enrichments
pub const AUDIO_SAMPLE_SIZE: usize = 16;

use silero_rs::vad_session::VadSession;

pub struct WhisperWrapper {
    pub ctx: WhisperContext,
    threads: u8,
    greedy_best_of: u8,
}

impl WhisperWrapper {
    pub fn new(threads: u8, greedy_best_of: u8, model_path: &str) -> Self {
        let whisper_ctx =
            WhisperContext::new_with_params(model_path, WhisperContextParameters::default());

        if whisper_ctx.is_err() {
            tracing::error!("failed to load model: {}", whisper_ctx.err().unwrap());
            panic!("failed to load model");
        }

        let whisper_ctx = whisper_ctx.unwrap();

        WhisperWrapper {
            ctx: whisper_ctx,
            threads,
            greedy_best_of,
        }
    }

    pub fn get_full_params(&self) -> FullParams {
        let mut params = FullParams::new(SamplingStrategy::Greedy {
            best_of: self.greedy_best_of as i32,
        });

        params.set_suppress_blank(true);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_progress(false);
        params.set_print_timestamps(false);
        params.set_token_timestamps(false);
        params.set_translate(true);
        params.set_n_threads(self.threads as i32);

        params
    }
}

pub fn push_audio_i16(
    params: FullParams,
    whisper_state: &mut WhisperState,
    timestamp_offset: i64, // this is the offset so we can keep track of the time
    sample_rate: u32,
    audio: &[i16],
    vad_session: &mut VadSession,
    whisper: &WhisperWrapper
) -> Vec<STTSegment> {
    
    let mut mut_timestamp_offset: i64 = timestamp_offset;
    let incoming_audio: fon::Audio<fon::chan::Ch16, 1> =
        fon::Audio::with_i16_buffer(sample_rate, audio);

    let mut converted_audio: fon::Audio<fon::chan::Ch32, 1> =
        fon::Audio::with_audio(AUDIO_SAMPLE_RATE as u32, &incoming_audio);
    
    let audio_vec: Vec<f32> = converted_audio.as_f32_slice().to_vec();

    let silero_results: Vec<silero_rs::vad_session::VoiceDetectResult> = vad_session.stateful_vad(audio_vec.clone(), sample_rate as i64).unwrap();
    let silero_result_len: i16 = silero_results.len() as i16;

    let mut stt_segments: Vec<Vec<STTSegment>> = Vec::new();
    let mut counter: i16 = 0;

    while counter < silero_result_len {

        let silero_result = silero_results.get(counter as usize).unwrap();
        tracing::debug!("Probability of speech: {}", silero_result.probability);
        if silero_result.voice_detect == false {
            
            let time_removed: f32 = (silero_result.end_period as f32) - (silero_result.start_period as f32);
            let time_removed_secs: f32 = time_removed / (sample_rate as f32);
            mut_timestamp_offset = mut_timestamp_offset + ((time_removed_secs * 100.0).round() as i64);
            
            counter += 1;
            continue;
        }

        counter += 1;
        let start_index = silero_result.start_period;
        let mut end_index = silero_result.end_period;

        while counter < silero_result_len {
            let silero_result = silero_results.get(counter as usize).unwrap();
            tracing::debug!("Probability of speech: {}", silero_result.probability);
            if silero_result.voice_detect == false {

                let time_removed: f32 = (silero_result.end_period as f32) - (silero_result.start_period as f32);
                let time_removed_secs: f32 = time_removed / (sample_rate as f32);
                mut_timestamp_offset = mut_timestamp_offset + ((time_removed_secs * 100.0).round() as i64);
                counter +=1;

                break;
            }

            end_index = silero_result.end_period;
            counter += 1;
        }

        end_index += 1;
        let audio_slice = &audio_vec[(start_index as usize)..(end_index as usize)];
        let stt_segment = push_audio_f32(whisper.get_full_params(), whisper_state, mut_timestamp_offset, audio_slice);
        stt_segments.push(stt_segment);
    }

    return stt_segments.into_iter().flatten().collect();
}

fn push_audio_f32(
    params: FullParams,
    whisper: &mut WhisperState,
    timestamp_offset: i64,
    audio: &[f32],
) -> Vec<STTSegment> {
    let stt_start_timestamp = std::time::Instant::now();

    let result = whisper.full(params, audio);
    if result.is_err() {
        tracing::error!("Error running STT: {:?}", result.err());
    }

    let mut fragments = vec![];

    // obtain segments from whisper
    let num_segments = whisper.full_n_segments();
    if num_segments.is_err() {
        tracing::error!("failed to get number of segments");
        return fragments;
    }
    let num_segments = num_segments.unwrap();

    for i in 0..num_segments {
        let segment = whisper.full_get_segment_text(i);
        if segment.is_err() {
            tracing::error!("failed to get segment");
            continue;
        }

        let start_timestamp = whisper.full_get_segment_t0(i);
        if start_timestamp.is_err() {
            tracing::error!("failed to get segment start timestamp");
            continue;
        }
        let end_timestamp = whisper.full_get_segment_t1(i);
        if end_timestamp.is_err() {
            tracing::error!("failed to get segment end timestamp");
            continue;
        }

        // create a new segment detail where audio was detected
        let fragment = STTSegment::new(
            start_timestamp.unwrap() + timestamp_offset,
            end_timestamp.unwrap() + timestamp_offset,
            segment.unwrap(),
        );

        tracing::trace!(
            "created fragment: {}, {}, {}",
            fragment.start_as_timestamp(),
            fragment.end_as_timestamp(),
            fragment.text()
        );

        fragments.push(fragment);
    }

    tracing::debug!(
        "STT processing of {} samples completed in {}mS",
        audio.len(),
        stt_start_timestamp.elapsed().as_millis()
    );

    fragments
}
