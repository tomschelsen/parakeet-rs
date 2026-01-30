use crate::config::PreprocessorConfig;
use crate::error::{Error, Result};
use hound::{WavReader, WavSpec};
use ndarray::Array2;
use non_empty_slice::NonEmptySlice;
use spectrograms::{
    audio::{MelDbSpectrogram, MelParams, SpectrogramParams, StftParams, WindowType},
};
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

/// Apply preemphasis to audio signal
///
/// # Arguments
/// * `audio` - Audio samples as f32 values
/// * `coef` - Preemphasis coefficient
///
/// # Returns
/// Preemphasized audio samples
pub fn apply_preemphasis(audio: &[f32], coef: f32) -> Vec<f32> {
    let mut result = Vec::with_capacity(audio.len());
    result.push(audio[0]);

    for i in 1..audio.len() {
        result.push(audio[i] - coef * audio[i - 1]);
    }

    result
}

/// Extract mel spectrogram features from raw audio samples.
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

    if channels > 1 {
        let mono: Vec<f32> = audio
            .chunks(channels as usize)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect();
        audio = mono;
    }

    // Apply preemphasis
    audio = apply_preemphasis(&audio, config.preemphasis);

    // Convert f32 to f64 for spectrograms crate
    let audio_f64: Vec<f64> = audio.iter().map(|&x| x as f64).collect();
    let audio_slice = NonEmptySlice::new(&audio_f64)
        .ok_or(Error::Audio("Empty audio data".to_string()))?;

    // Create spectrogram parameters
    let stft_params = StftParams::new(
        config.n_fft,
        config.hop_length,
        WindowType::Hanning,
        true, // center
    )
    .map_err(|e| Error::Audio(format!("STFT params error: {e}")))?;

    let spectrogram_params = SpectrogramParams::new(stft_params, config.sampling_rate as f64)
        .map_err(|e| Error::Audio(format!("Spectrogram params error: {e}")))?;

    // Create mel filterbank parameters
    let mel_params = MelParams::new(
        config.feature_size,
        0.0, // f_min
        config.sampling_rate as f64 / 2.0, // f_max (Nyquist)
    )
    .map_err(|e| Error::Audio(format!("Mel params error: {e}")))?;

    // Compute mel spectrogram in dB scale
    let mel_spec = MelDbSpectrogram::compute(&audio_slice, &spectrogram_params, &mel_params, None)
        .map_err(|e| Error::Audio(format!("Mel spectrogram computation failed: {e}")))?;

    // Convert from f64 to f32 and normalize
    let spec_data = mel_spec.data();
    let (rows, cols) = (spec_data.nrows(), spec_data.ncols());
    let mut mel_spectrogram = Array2::zeros((cols, rows));

    for i in 0..rows {
        for j in 0..cols {
            mel_spectrogram[[j, i]] = spec_data[[i, j]] as f32;
        }
    }

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