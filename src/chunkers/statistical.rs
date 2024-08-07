use std::{cmp::max, os::linux::raw};

use crate::embeddings::{cloud::openai::OpenAIEmbeder, embed::TextEmbed, local::jina::JinaEmbeder, normalize_l2};
use candle_core::Tensor;
use itertools::{enumerate, Itertools};
use text_splitter::{ChunkConfig, ChunkSizer, TextSplitter};
use tokenizers::{tokenizer, Tokenizer};

#[derive(Debug)]
pub struct StatisticalChunker<T: TextEmbed, Sizer: ChunkSizer> {
    pub encoder: T,
    pub splitter: TextSplitter<Sizer>,
    pub score_threshold: f32,
    pub device: candle_core::Device,
    pub threshold_adjustment: f32,
    pub dynamic_threshold: bool,
    pub window_size: usize,
    pub min_split_tokens: usize,
    pub max_split_tokens: usize,
    pub split_token_tolerance: usize,
    pub tokenizer: Tokenizer,
}
impl Default for StatisticalChunker<JinaEmbeder, Tokenizer> {
    fn default() -> Self {
        let tokenizer = Tokenizer::from_pretrained("BEE-spoke-data/cl100k_base-mlm", None).unwrap();

        let splitter = TextSplitter::new(ChunkConfig::new(200).with_sizer(tokenizer.clone()));
        let encoder = JinaEmbeder::default();
        let score_threshold = 0.9;
        let device = candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu);
        Self {
            encoder,
            splitter,
            score_threshold,
            device,
            threshold_adjustment: 0.01,
            dynamic_threshold: true,
            window_size: 5,
            min_split_tokens: 200,
            max_split_tokens: 400,
            split_token_tolerance: 10,
            tokenizer,
        }
    }
}

impl<T: TextEmbed, Sizer: ChunkSizer> StatisticalChunker<T, Sizer> {
    pub fn new(
        encoder: T,
        splitter: TextSplitter<Sizer>,
        score_threshold: f32,
        threshold_adjustment: f32,
        dynamic_threshold: bool,
        window_size: usize,
        min_split_tokens: usize,
        max_split_tokens: usize,
        split_token_tolerance: usize,
        tokenizer: Tokenizer,
    ) -> Self {
        Self {
            encoder,
            splitter,
            score_threshold,
            device: candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu),
            threshold_adjustment,
            dynamic_threshold,
            window_size,
            min_split_tokens,
            max_split_tokens,
            split_token_tolerance,
            tokenizer,
        }
    }
    pub fn _chunk(&self, text: &str, batch_size: usize) -> Vec<String> {
        let splits = self
            .splitter
            .chunks(text)
            .collect::<Vec<_>>()
            .into_iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>();
        let mut chunks: Vec<String> = Vec::new();
        let mut last_chunk: String = "".to_string();

        for i in &(0..splits.len()).chunks(batch_size) {
            let indices = i.collect::<Vec<_>>();
            let mut batch_splits = indices
                .into_iter()
                .map(|idx| splits[idx].to_string())
                .collect::<Vec<_>>();

            if !last_chunk.is_empty() {
                batch_splits = vec![last_chunk.clone()]
                    .into_iter()
                    .chain(batch_splits)
                    .collect::<Vec<_>>();
            }

            let encoded_splits = self.encoder.embed(&batch_splits, Some(32)).unwrap();
            let similarities = self._calculate_similarity_scores(&encoded_splits);
            let calculated_threshold: f32;
            if self.dynamic_threshold {
                calculated_threshold = self._find_optimal_threshold(&batch_splits, &similarities);
            } else {
                calculated_threshold = self.score_threshold;
            }

            let split_indices = self._find_split_indices(&similarities, calculated_threshold);
            let doc_chunks:Vec<String> = self._split_documents(batch_splits, split_indices, similarities);

            if doc_chunks.len() > 1{
               
               //add doc chunks to chunks
                chunks.extend(doc_chunks.clone().into_iter());
                // last chunk is last element of doc_chunks
                last_chunk =  doc_chunks.last().unwrap().to_string();
            }else{
                last_chunk = doc_chunks[0].clone();

                
            }
        }
        if !last_chunk.is_empty(){
            chunks.push(last_chunk);
        }

        for chunk in chunks.iter(){
            println!("-----Chunk---\n{}", chunk);
        }
        chunks
    }

    fn _calculate_similarity_scores(&self, encoded_splits: &Vec<Vec<f32>>) -> Vec<f32> {
        let embed_dim = encoded_splits[0].len();
        let mut raw_similarities: Vec<f32> = Vec::new();
        let encoded_splits_tensor = Tensor::from_vec(
            encoded_splits
                .into_iter()
                .flatten()
                .map(|&x| x)
                .collect::<Vec<f32>>(),
            (encoded_splits.len(), embed_dim),
            &self.device,
        )
        .unwrap();

        for i in 1..encoded_splits.len() {
            let window_start = max(0, i as isize - self.window_size as isize) as usize;
            let indexes = Tensor::arange(window_start as i64, i as i64, &self.device).unwrap();
            let encoded_splits_window = encoded_splits_tensor.index_select(&indexes, 0).unwrap();

            let cumulative_context = encoded_splits_window.mean_keepdim(0).unwrap();
            let cumulative_context_norm = cumulative_context
                .sqr()
                .unwrap()
                .get(0)
                .unwrap()
                .sum(0)
                .unwrap()
                .sqrt()
                .unwrap();
            let encoded_splits_tensor_norm = encoded_splits_tensor
                .get(i)
                .unwrap()
                .sqr()
                .unwrap()
                .sum(0)
                .unwrap()
                .sqrt()
                .unwrap();
            let norm = (encoded_splits_tensor_norm * cumulative_context_norm).unwrap();
            let curr_sim_score = encoded_splits_tensor
                .get(0)
                .unwrap()
                .reshape((1, embed_dim))
                .unwrap()
                .matmul(&cumulative_context.transpose(0, 1).unwrap())
                .unwrap()
                .squeeze(1)
                .unwrap();

            let curr_sim_score_scaled = curr_sim_score
                .broadcast_div(&norm)
                .unwrap()
                .get(0)
                .unwrap()
                .to_vec0::<f32>()
                .unwrap();
            raw_similarities.push(curr_sim_score_scaled);
        }
        raw_similarities
    }

    fn _find_optimal_threshold(&self, batch_splits: &Vec<String>, similarities: &Vec<f32>) -> f32 {
        let tokens = self
            .tokenizer
            .encode_batch(batch_splits.to_vec(), true)
            .unwrap();
        let token_counts = tokens
            .iter()
            .map(|tokens| tokens.get_ids().len())
            .collect::<Vec<_>>();

        let cumulative_token_counts = token_counts
            .iter()
            .scan(0, |state, &x| {
                *state += x;
                Some(*state)
            })
            .collect::<Vec<_>>();

        // Add an extra element to cumulative_token_counts to handle the end case
        let mut cumulative_token_counts = cumulative_token_counts;
        cumulative_token_counts.push(cumulative_token_counts.last().copied().unwrap_or(0));

        // analyze the distribution of similarity scores to oset initial bounds
        let median_score = statistical::median(similarities);
        let std_dev = statistical::standard_deviation(similarities, None);

        // set initial bounds based on median and standard deviation
        let mut low = f32::max(0.0, median_score - std_dev);
        let mut high = f32::min(1.0, median_score + std_dev);

        let mut iteration = 0;
        let mut median_tokens: usize;
        let mut calculated_threshold = 0.0;

        while low <= high {
            println!("Iteration: {}", iteration);
            calculated_threshold = (low + high) / 2.0;
            let split_indices = self._find_split_indices(&similarities, calculated_threshold);
            let split_token_counts: Vec<usize> = [0]
                .iter()
                .chain(split_indices.iter())
                .zip(
                    split_indices
                        .iter()
                        .chain(std::iter::once(&token_counts.len())),
                )
                .map(|(start, end)| cumulative_token_counts[*end] - cumulative_token_counts[*start])
                .collect();

            median_tokens = statistical::median(&split_token_counts);

            if self.min_split_tokens - self.split_token_tolerance <= median_tokens
                && median_tokens <= self.max_split_tokens + self.split_token_tolerance
            {
                break;
            } else if median_tokens < self.min_split_tokens {
                high = calculated_threshold - self.threshold_adjustment;
            } else {
                low = calculated_threshold + self.threshold_adjustment;
            }
            iteration += 1;
        }
        calculated_threshold
    }
    fn _find_split_indices(&self, similarities: &Vec<f32>, threshold: f32) -> Vec<usize> {
        let mut split_indices = Vec::new();
        for (idx, score) in enumerate(similarities) {
            if score.clone() < threshold {
                split_indices.push(idx + 1);
            }
        }
        split_indices
    }

    fn _split_documents(
        &self,
        docs: Vec<String>,
        split_indices: Vec<usize>,
        similarities: Vec<f32>,
    ) -> Vec<String> {
        let tokens = self.tokenizer.encode_batch(docs.to_vec(), true).unwrap();
        let token_counts = tokens
            .iter()
            .map(|tokens| tokens.get_ids().len())
            .collect::<Vec<_>>();
        let mut chunks: Vec<String> = Vec::new();
        let mut current_split = Vec::new();
        let mut current_tokens_count = 0;

        for (doc_idx, doc) in enumerate(docs) {
            let doc_token_count = token_counts[doc_idx];

            if split_indices.contains(&(doc_idx + 1)) {
                if self.min_split_tokens <= current_tokens_count + doc_token_count
                    && current_tokens_count + doc_token_count <= self.max_split_tokens
                {
                    current_split.push(doc);
                    current_tokens_count += doc_token_count;
                }
                chunks.push(current_split.join(" "));   
                current_split = Vec::new();
                current_tokens_count = 0;
                continue;
            }
            if current_tokens_count + doc_token_count > self.max_split_tokens{
                if current_tokens_count>=self.min_split_tokens{
                    chunks.push(current_split.join(" "));

                }
                current_split = Vec::new();
                current_tokens_count = 0;
            }
            current_split.push(doc);
            current_tokens_count += doc_token_count;
        }

        if !current_split.is_empty() {
            chunks.push(current_split.join(" "));
        }

        chunks

    }
}

