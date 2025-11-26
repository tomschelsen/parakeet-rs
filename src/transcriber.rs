use crate::audio::{load_audio, SamplesAndMetadata};
use crate::decoder::TranscriptionResult;
use crate::error::{Error, Result};
use crate::timestamps::TimestampMode;
use std::path::Path;

pub trait Batcher {
    fn build_batch(&self) -> Result<Vec<SamplesAndMetadata>>;
}

impl<P> Batcher for P
where
    P: AsRef<Path>,
{
    fn build_batch(&self) -> Result<Vec<SamplesAndMetadata>> {
        let loaded = load_audio(self)?;
        Ok(Vec::from([loaded]))
    }
}

impl<P> Batcher for [P]
where
    P: AsRef<Path>,
{
    fn build_batch(&self) -> Result<Vec<SamplesAndMetadata>> {
        let draft: Vec<SamplesAndMetadata> =
            self.iter().filter_map(|x| load_audio(x).ok()).collect();
        if !draft.is_empty() {
            Ok(draft)
        } else {
            Err(Error::Audio("No file could be properly loaded".into()))
        }
    }
}

pub trait Transcriber {
    fn transcribe_samples(
        &mut self,
        audio: Vec<f32>,
        sample_rate: u32,
        channels: u16,
        mode: Option<TimestampMode>,
    ) -> Result<TranscriptionResult>;

    fn transcribe<P: Batcher>(
        &mut self,
        input: P,
        mode: Option<TimestampMode>,
    ) -> Result<Vec<TranscriptionResult>> {
        let batch = input.build_batch()?;
        let res: Vec<TranscriptionResult> = batch
            .into_iter()
            .filter_map(|f| {
                self.transcribe_samples(f.samples, f.spec.sample_rate, f.spec.channels, mode)
                    .ok()
            })
            .collect();
        Ok(res)
    }
}
