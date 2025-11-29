use crate::audio::load_audio;
use crate::config::PreprocessorConfig;
use crate::decoder::TranscriptionResult;
use crate::error::Result;
use crate::timestamps::{process_timestamps, TimestampMode};
use std::path::{Path, PathBuf};

pub trait OneOrManyPaths {
    fn normalize(&self) -> Vec<PathBuf>;
}

impl<P> OneOrManyPaths for P
where
    P: AsRef<Path>,
{
    fn normalize(&self) -> Vec<PathBuf> {
        Vec::from([self.as_ref().to_path_buf()])
    }
}

impl<P> OneOrManyPaths for [P]
where
    P: AsRef<Path>,
{
    fn normalize(&self) -> Vec<PathBuf> {
        self.iter().map(|p| p.as_ref().to_path_buf()).collect()
    }
}

pub trait Transcriber {
    fn preprocessor_config(&self) -> &PreprocessorConfig;

    fn post_process_trancription_result(
        mut result: TranscriptionResult,
        mode: Option<TimestampMode>,
    ) -> TranscriptionResult {
        // Apply timestamp mode conversion
        let mode = mode.unwrap_or(TimestampMode::Tokens);
        result.tokens = process_timestamps(&result.tokens, mode);

        // Rebuild full text from processed tokens
        result.text = result
            .tokens
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        result
    }

    fn transcribe_16khz_mono_samples(
        &mut self,
        audio: Vec<f32>,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult>;

    fn transcribe<P: OneOrManyPaths>(
        &mut self,
        paths: P,
        mode: Option<TimestampMode>,
    ) -> Result<Vec<TranscriptionResult>> {
        let file_batch = paths.normalize();
        let checked_loaded_audio = file_batch
            .iter()
            .map(|f| load_audio(f, self.preprocessor_config()))
            .collect::<Result<Vec<_>>>()?;

        // TODO : pass a Vec<Vec<32>> to transcribe_..._samples instead
        checked_loaded_audio
            .into_iter()
            .map(|audio| self.transcribe_16khz_mono_samples(audio, mode))
            .collect::<Result<Vec<_>>>()
    }
}
