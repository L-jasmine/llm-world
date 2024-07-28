use simple_llama::{
    llm::{LlamaCtx, SimpleOption},
    Content,
};

use crate::chat::im_channel::{Message, MessageRx, MessageTx};

#[derive(Debug, Clone)]
pub enum Token {
    Start,
    Chunk(String),
    End(String),
}

pub struct LocalLlama {
    ctx: LlamaCtx,
    prompts: Vec<Content>,
    rx: MessageRx,
    tx: MessageTx,
}

impl LocalLlama {
    pub fn new(ctx: LlamaCtx, prompts: Vec<Content>, rx: MessageRx, tx: MessageTx) -> Self {
        LocalLlama {
            ctx,
            prompts,
            rx,
            tx,
        }
    }

    fn wait_input(&mut self) -> anyhow::Result<()> {
        loop {
            let message = self.rx.recv()?;
            match message {
                Message::GenerateByUser(user) => {
                    self.prompts.push(user);
                    self.prompts.push(Content {
                        role: simple_llama::Role::Assistant,
                        message: String::new(),
                    });
                }
                Message::Generate(assistant) => match self.prompts.last_mut() {
                    Some(content) => *content = assistant,
                    None => {
                        self.prompts.push(assistant);
                    }
                },
                Message::Assistant(_) => {
                    continue;
                }
            }
            break Ok(());
        }
    }

    pub fn run_loop(&mut self) -> anyhow::Result<()> {
        loop {
            self.wait_input()?;

            self.tx.send(Message::Assistant(Token::Start))?;
            let mut stream = self.ctx.chat(&self.prompts, SimpleOption::Temp(0.9))?;

            for token in &mut stream {
                self.tx.send(Message::Assistant(Token::Chunk(token)))?;
            }

            let message: String = stream.into();
            self.prompts
                .last_mut()
                .map(|c| c.message.push_str(&message));
            let last_message = self
                .prompts
                .last()
                .map(|c| c.message.clone())
                .unwrap_or_default();

            self.tx.send(Message::Assistant(Token::End(last_message)))?;
        }
    }

    pub fn filter(message: &Message) -> Option<Message> {
        if matches!(message, Message::Assistant(..)) {
            None
        } else {
            Some(message.clone())
        }
    }
}
