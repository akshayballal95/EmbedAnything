use std::{cell::RefCell, collections::HashMap, sync::RwLock};

use anyhow::Error as E;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::{colpali::Model, paligemma};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

use crate::embeddings::embed::{EmbedData, EmbedImage, EmbeddingResult};

pub struct ColPaliEmbedder {
    pub model: RwLock<Model>,
    pub tokenizer: Tokenizer,
    pub config: paligemma::Config,
    dtype: DType,
    dummy_input: Tensor,
}

impl ColPaliEmbedder {
    pub fn new(model_id: &str, revision: Option<&str>) -> Result<Self, anyhow::Error> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo: hf_hub::api::sync::ApiRepo = match revision {
            Some(rev) => api.repo(hf_hub::Repo::with_revision(
                model_id.to_string(),
                hf_hub::RepoType::Model,
                rev.to_string(),
            )),
            None => api.repo(hf_hub::Repo::new(
                model_id.to_string(),
                hf_hub::RepoType::Model,
            )),
        };

        let tokenizer_api = api.repo(hf_hub::Repo::new(
            "vidore/colpali".to_string(),
            hf_hub::RepoType::Model,
        ));

        let (tokenizer_filename, weights_filename) = {
            let tokenizer = tokenizer_api.get("tokenizer.json")?;
            let weights = hub_load_safetensors(&repo, "model.safetensors.index.json")?;

            (tokenizer, weights)
        };

        // let config = std::fs::read_to_string(config_filename)?;
        // let config: paligemma::Config = serde_json::from_str(&config)?;
        let config: paligemma::Config = paligemma::Config::paligemma_3b_448();

        let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        let trunc = TruncationParams {
            strategy: tokenizers::TruncationStrategy::LongestFirst,
            max_length: config.text_config.max_position_embeddings as usize,
            ..Default::default()
        };

        tokenizer
            .with_padding(Some(pp))
            .with_truncation(Some(trunc))
            .unwrap();

        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights_filename, dtype, &device)? };

        let model = Model::new(&config, vb)?;
        let dummy_prompt: &str = "Describe the image";

        let dummy_input = tokenize_batch(&tokenizer, vec![dummy_prompt], &device)?;

        Ok(Self {
            model: RwLock::new(model),
            tokenizer,
            config,
            dtype,
            dummy_input,
        })
    }

    pub fn load_image<T: AsRef<std::path::Path>>(
        &self,
        path: T,
        image_size: usize,
    ) -> anyhow::Result<Tensor> {
        let img = image::ImageReader::open(path)?.decode()?;
        let (height, width) = (image_size, image_size);
        let img = img.resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::Triangle,
        );

        let img = img.to_rgb8();

        let img = img.into_raw();
        let img = Tensor::from_vec(
            img,
            (height, width, 3),
            &Device::cuda_if_available(0).unwrap_or(Device::Cpu),
        )?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?;
        // .unsqueeze(0)?;
        Ok(img)
    }

    fn load_images<T: AsRef<std::path::Path>>(
        &self,
        paths: &[T],
        image_size: usize,
    ) -> anyhow::Result<Tensor> {
        let mut images = vec![];

        for path in paths {
            let tensor = self.load_image(path, image_size)?;
            images.push(tensor);
        }

        let images = Tensor::stack(&images, 0)?;

        Ok(images)
    }

    pub fn embed(
        &self,
        text_batch: &[String],
        batch_size: Option<usize>,
    ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
        let mut encodings = Vec::new();
        for mini_text_batch in text_batch.chunks(batch_size.unwrap_or(32)) {
            let input_ids = tokenize_batch(
                &self.tokenizer,
                mini_text_batch
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>(),
                &Device::cuda_if_available(0).unwrap_or(Device::Cpu),
            )?;
            let batch_encodings = self
                .model
                .write()
                .unwrap()
                .forward_text(&input_ids)?
                .to_dtype(DType::F32)?;

            encodings.extend(
                batch_encodings
                    .to_vec3::<f32>()?
                    .iter()
                    .map(|x| EmbeddingResult::Sparse(x.to_vec())),
            );
        }
        Ok(encodings)
    }
}

impl EmbedImage for ColPaliEmbedder {
    fn embed_image<T: AsRef<std::path::Path>>(
        &self,
        image_path: T,
        metadata: Option<HashMap<String, String>>,
    ) -> anyhow::Result<EmbedData> {
        let pixel_values = self
            .load_image(image_path, self.config.vision_config.image_size)?
            .unsqueeze(0)?
            .to_dtype(self.dtype)?;
        let encoding = self
            .model
            .write()
            .unwrap()
            .forward_images(&pixel_values, &self.dummy_input)?
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?
            .iter()
            .map(|x| EmbeddingResult::Sparse(x.to_vec()))
            .collect::<Vec<_>>();

        Ok(EmbedData::new(encoding[0].clone(), None, metadata.clone()))
    }

    fn embed_image_batch<T: AsRef<std::path::Path>>(
        &self,
        image_paths: &[T],
    ) -> anyhow::Result<Vec<EmbedData>> {
        let pixel_values = self
            .load_images(image_paths, self.config.vision_config.image_size)?
            .to_dtype(self.dtype)?;
        let encodings = self
            .model
            .write()
            .unwrap()
            .forward_images(&pixel_values, &self.dummy_input)?
            .to_dtype(DType::F32)?
            .to_vec3::<f32>()?;

        Ok(encodings
            .iter()
            .map(|x| EmbedData::new(EmbeddingResult::Sparse(x.to_vec()), None, None))
            .collect::<Vec<_>>())
    }
}

fn tokenize_batch(
    tokenizer: &Tokenizer,
    text_batch: Vec<&str>,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let tokens = tokenizer.encode_batch(text_batch, true).map_err(E::msg)?;
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<candle_core::Result<Vec<_>>>()?;

    Ok(Tensor::stack(&token_ids, 0)?)
}

pub fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>, E> {
    let json_file = repo.get(json_file).map_err(candle_core::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => anyhow::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => anyhow::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(candle_core::Error::wrap))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(safetensors_files)
}
