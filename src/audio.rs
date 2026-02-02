use crate::config::PreprocessorConfig;
use crate::error::{Error, Result};
use hound::{WavReader, WavSpec};
use ndarray::Array2;
use spectrograms::{MelParams, SpectrogramParamsBuilder, WindowType};
use std::num::NonZeroUsize;
use std::path::Path;

pub fn load_audio<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, WavSpec)> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Audio(format!("Failed to read float samples: {e}")))?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Audio(format!("Failed to read int samples: {e}")))?,
    };

    Ok((samples, spec))
}

/// Extract mel spectrogram features from raw audio samples using the spectrograms crate.
///
/// # Arguments
///
/// * `audio` - Audio samples as f32 values
/// * `sample_rate` - Sample rate in Hz
/// * `channels` - Number of audio channels
/// * `config` - Preprocessor configuration
///
/// # Returns
///
/// 2D array of mel spectrogram features (time_steps x feature_size)
pub fn extract_features_raw(
    mut audio: Vec<f32>,
    sample_rate: u32,
    channels: u16,
    config: &PreprocessorConfig,
) -> Result<Array2<f32>> {
    if sample_rate != config.sampling_rate as u32 {
        return Err(Error::Audio(format!(
            "Audio sample rate {} doesn't match expected {}. Please resample your audio first.",
            sample_rate, config.sampling_rate
        )));
    }

    // Convert to mono if multi-channel
    if channels > 1 {
        let mono: Vec<f32> = audio
            .chunks(channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect();
        audio = mono;
    }

    // Convert audio samples to f64 for spectrograms crate
    let audio_f64: Vec<f64> = audio.into_iter().map(|x| x as f64).collect();
    let audio_slice = non_empty_slice::NonEmptySlice::new(&audio_f64)
        .expect("Audio data should not be empty");

    // Create spectrogram parameters
    let spectrogram_params = SpectrogramParamsBuilder::default()
        .sample_rate(config.sampling_rate as f64)
        .n_fft(NonZeroUsize::new(config.n_fft).ok_or_else(|| {
            Error::Audio(format!("n_fft must be non-zero, got {}", config.n_fft))
        })?)
        .hop_size(NonZeroUsize::new(config.hop_length).ok_or_else(|| {
            Error::Audio(format!("hop_length must be non-zero, got {}", config.hop_length))
        })?)
        .window(WindowType::Hanning)
        .centre(true)
        .build()
        .map_err(|e| Error::Audio(format!("Failed to build spectrogram params: {e}")))?;

    // Create mel filterbank parameters
    let mel_params = MelParams::new(
        NonZeroUsize::new(config.feature_size).ok_or_else(|| {
            Error::Audio(format!("feature_size must be non-zero, got {}", config.feature_size))
        })?,
        0.0, // f_min
        (config.sampling_rate / 2) as f64, // f_max
    )
    .map_err(|e| Error::Audio(format!("Failed to create mel params: {e}")))?;

    // Compute mel spectrogram using spectrograms crate
    let spectrogram = spectrograms::MelPowerSpectrogram::compute(
        audio_slice.as_ref(),
        &spectrogram_params,
        &mel_params,
        None, // No additional log params needed
    )
    .map_err(|e| Error::Audio(format!("Failed to compute mel spectrogram: {e}")))?;

    // Convert to ndarray Array2<f32> and transpose to match expected format (time_steps x feature_size)
    let mut mel_spectrogram = spectrogram.data().mapv(|x| x as f32).reversed_axes();

    // Normalize each feature dimension to mean=0, std=1
    let num_frames = mel_spectrogram.shape()[0];
    let num_features = mel_spectrogram.shape()[1];

    for feat_idx in 0..num_features {
        let mut column = mel_spectrogram.column_mut(feat_idx);
        let mean: f32 = column.iter().sum::<f32>() / num_frames as f32;
        let variance: f32 =
            column.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / num_frames as f32;
        let std = variance.sqrt().max(1e-10);

        for val in column.iter_mut() {
            *val = (*val - mean) / std;
        }
    }

    Ok(mel_spectrogram)
}
