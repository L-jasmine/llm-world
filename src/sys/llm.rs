use std::{fmt::Display, sync::Arc};

use llama_cpp_2::{
    context::LlamaContext,
    llama_backend::LlamaBackend,
    llama_batch::LlamaBatch,
    model::{self, LlamaModel, Special},
    token::data_array::LlamaTokenDataArray,
};

pub use llama_cpp_2::context::params::LlamaContextParams;
pub use llama_cpp_2::model::params::LlamaModelParams;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Role {
    #[serde(rename = "system")]
    System,
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
}

impl Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let role = self.as_ref();
        write!(f, "{role}")
    }
}

impl AsRef<str> for Role {
    fn as_ref(&self) -> &str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Content {
    pub role: Role,
    pub message: String,
}

impl AsRef<Content> for Content {
    fn as_ref(&self) -> &Content {
        self
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub enum SimpleOption {
    None,
    Temp(f32),
    TopP(f32, usize),
    TopK(i32, usize),
    MirostatV2(f32, f32),
}

impl Default for SimpleOption {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PromptTemplate {
    pub header_prefix: String,
    pub header_suffix: String,
    pub end_of_content: String,
    pub stops: Vec<String>,
}

impl PromptTemplate {
    fn encode_string<I: Iterator<Item = C>, C: AsRef<Content>>(&self, content: I) -> String {
        let mut result = String::with_capacity(128);
        // let len = content.count();
        let mut last_role = Role::System;
        for c in content {
            let c = c.as_ref();
            last_role = c.role.clone();
            if !result.is_empty() {
                // last content end
                result.push_str(&self.end_of_content);
            }
            result.push_str(&self.header_prefix);
            result.push_str(&c.role.to_string());
            result.push_str(&self.header_suffix);
            result.push_str(&c.message);
        }

        match last_role {
            Role::Assistant => {}
            _ => {
                result.push_str(&self.end_of_content);
                result.push_str(&self.header_prefix);
                result.push_str("assistant");
                result.push_str(&self.header_suffix);
            }
        }

        log::debug!("prompts:\n{}", result);

        result
    }

    fn post_handle_content(&self, content: &mut String) -> bool {
        let bs = unsafe { content.as_mut_vec() };
        let len = bs.len();

        let mut s = false;
        for stop in &self.stops {
            let stop_bs = stop.as_bytes();

            if bs.ends_with(stop_bs) {
                bs.truncate(len - stop_bs.len());
                s = true;
                break;
            }
        }
        s
    }
}

#[allow(unused)]
pub struct LlmModel {
    pub model_path: String,
    pub model: LlamaModel,
    pub model_params: LlamaModelParams,
    pub backend: LlamaBackend,
    pub prompt_template: PromptTemplate,
}

impl LlmModel {
    pub fn new(
        model_path: String,
        model_params: LlamaModelParams,
        prompt_template: PromptTemplate,
    ) -> llama_cpp_2::Result<Arc<Self>> {
        let backend = LlamaBackend::init()?;
        let llama = LlamaModel::load_from_file(&backend, &model_path, &model_params)?;
        let model = Self {
            model_path,
            model: llama,
            model_params,
            backend,
            prompt_template,
        };

        Ok(Arc::new(model))
    }
}

pub struct LlamaCtx {
    decoder: encoding_rs::Decoder,
    ctx: LlamaContext<'static>,
    batch: LlamaBatch,
    model: Arc<LlmModel>,
    n_cur: usize,
}

impl LlamaCtx {
    pub fn new(model: Arc<LlmModel>, ctx_params: LlamaContextParams) -> anyhow::Result<Self> {
        let ctx = model.model.new_context(&model.backend, ctx_params)?;
        let n_tokens = ctx.n_batch();
        let ctx = unsafe { std::mem::transmute(ctx) };
        let batch = LlamaBatch::new(n_tokens as usize, 1);
        let decoder = encoding_rs::UTF_8.new_decoder();

        Ok(Self {
            decoder,
            ctx,
            model,
            batch,
            n_cur: 0,
        })
    }

    pub fn chat<'a, I: IntoIterator<Item = C>, C: AsRef<Content>>(
        &'a mut self,
        prompts: I,
        simple_option: SimpleOption,
    ) -> anyhow::Result<LlamaModelChatStream<Self>> {
        self.decoder = encoding_rs::UTF_8.new_decoder();

        self.reset_batch_with_prompt(prompts.into_iter())?;

        let mut mu = 0.;
        if let SimpleOption::MirostatV2(tau, _) = &simple_option {
            mu = *tau * 2.0;
        }

        Ok(LlamaModelChatStream {
            llama_ctx: self,
            simple_option,
            mu,
        })
    }

    fn reset_batch_with_prompt<I: Iterator<Item = C>, C: AsRef<Content>>(
        &mut self,
        prompts: I,
    ) -> anyhow::Result<()> {
        self.ctx.clear_kv_cache();
        self.batch.clear();
        self.n_cur = 0;

        let tokens = self.model.model.str_to_token(
            &self.model.prompt_template.encode_string(prompts),
            model::AddBos::Always,
        )?;

        let last_index = (tokens.len() - 1) as i32;
        let n_tokens = self.ctx.n_batch();

        for (i, token) in (0_i32..).zip(tokens.into_iter()) {
            let is_last = i == last_index;

            self.batch.add(token, self.n_cur as i32, &[0], is_last)?;
            self.n_cur += 1;

            if !is_last && self.batch.n_tokens() == n_tokens as i32 {
                self.ctx.decode(&mut self.batch)?;
                self.batch.clear();
            }
        }

        Ok(())
    }

    fn take_a_token(
        &mut self,
        simple_option: SimpleOption,
        mu: &mut f32,
    ) -> anyhow::Result<Option<String>> {
        self.ctx.decode(&mut self.batch)?;

        let candidates = self.ctx.candidates_ith(self.batch.n_tokens() - 1);
        let mut candidates_p = LlamaTokenDataArray::from_iter(candidates, false);
        let new_token_id = match simple_option {
            SimpleOption::None => candidates_p.sample_token(&mut self.ctx),
            SimpleOption::Temp(temperature) => {
                candidates_p.sample_temp(None, temperature);
                candidates_p.sample_token(&mut self.ctx)
            }
            SimpleOption::TopP(p, min_keep) => {
                candidates_p.sample_top_p(None, p, min_keep);
                candidates_p.sample_token(&mut self.ctx)
            }
            SimpleOption::TopK(k, min_keep) => {
                candidates_p.sample_top_k(None, k, min_keep);
                candidates_p.sample_token(&mut self.ctx)
            }
            SimpleOption::MirostatV2(tau, eta) => {
                candidates_p.sample_token_mirostat_v2(&mut self.ctx, tau, eta, mu)
            }
        };

        self.batch.clear();
        self.batch
            .add(new_token_id, self.n_cur as i32, &[0], true)?;
        self.n_cur += 1;

        if new_token_id == self.model.model.token_eos() {
            return Ok(None);
        } else {
            let output_bytes = self
                .model
                .model
                .token_to_bytes(new_token_id, Special::Tokenize)?;
            let mut output_string = String::with_capacity(32);
            let _decode_result =
                self.decoder
                    .decode_to_string(&output_bytes, &mut output_string, false);

            Ok(Some(output_string))
        }
    }
}

pub struct LlamaModelChatStream<'a, CTX> {
    llama_ctx: &'a mut CTX,
    simple_option: SimpleOption,
    mu: f32,
}

impl<'a> LlamaModelChatStream<'a, LlamaCtx> {
    pub fn next_token(&mut self) -> anyhow::Result<Option<String>> {
        self.llama_ctx
            .take_a_token(self.simple_option, &mut self.mu)
    }

    pub fn is_stop(&self, content: &mut String) -> bool {
        self.llama_ctx
            .model
            .prompt_template
            .post_handle_content(content)
    }
}
