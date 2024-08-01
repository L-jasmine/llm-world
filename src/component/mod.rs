use std::{collections::LinkedList, time::Duration};

use chat::{Input, Output};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Layout},
    widgets::{Block, Paragraph, Tabs},
    Frame, Terminal,
};
use simple_llama::{Content, LlamaCtx, LlamaModelChatStream};

pub mod chat;
pub mod lab;

#[derive(Debug, Clone)]
pub enum Token {
    Chunk(String),
    End(String),
}

pub struct App {
    pub select_tabs: usize,
    pub exit_n: u8,
    pub chat: chat::ChatComponent,
    pub lab: lab::Lab,
    pub prompts_path: String,
}

impl App {
    pub fn new(prompts_path: String) -> Self {
        Self {
            chat: chat::ChatComponent::new(),
            lab: lab::Lab {
                prompts_path: prompts_path.clone(),
                messages: chat::MessagesComponent::new(),
            },
            select_tabs: 0,
            exit_n: 0,
            prompts_path,
        }
    }

    pub fn render(&mut self, contents: &LinkedList<Content>, f: &mut Frame) {
        let vertical = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(3),
            Constraint::Length(1),
            Constraint::Length(1),
        ]);

        let [tabs_area, main_area, help_area, event_area] = vertical.areas(f.size());

        let tabs = Tabs::new(vec!["Chat", "Lab"])
            .select(self.select_tabs)
            .padding("[", "]")
            .block(Block::bordered());

        f.render_widget(tabs, tabs_area);
        match self.select_tabs {
            0 => self.chat.render(contents, f, main_area),
            _ => self.lab.render(contents, f, main_area),
        }

        let help_message = Paragraph::new(format!("help: [Ctrl+R rewrite] [Esc+Esc quit]"));
        f.render_widget(help_message, help_area);

        let help_message = Paragraph::new(format!("{}", self.chat.event));
        f.render_widget(help_message, event_area);
    }

    pub fn handler_input(
        &mut self,
        input: Input,
        contents: &mut LinkedList<Content>,
    ) -> anyhow::Result<Output> {
        let last_exit_n = self.exit_n;
        if matches!(input, Input::Event(..)) {
            self.exit_n = 0;
        }
        match input {
            Input::Event(Event::Key(event)) if event.code == KeyCode::Tab => {
                self.select_tabs = (self.select_tabs + 1) % 2;
                Ok(Output::Normal)
            }
            Input::Event(Event::Key(input)) if input.code == KeyCode::Esc => {
                self.exit_n = 1;
                if last_exit_n != 0 {
                    Ok(Output::Exit)
                } else {
                    Ok(Output::Normal)
                }
            }
            Input::Message(token) => match token {
                Token::End(end) => {
                    if let Some(c) = contents.back_mut() {
                        c.message = end;
                    }
                    Ok(Output::Normal)
                }
                Token::Chunk(chunk) => {
                    if let Some(c) = contents.back_mut() {
                        c.message.push_str(&chunk);
                    }
                    Ok(Output::Normal)
                }
            },

            input => match self.select_tabs {
                0 => Ok(self.chat.handler_input(input, contents)),
                _ => self.lab.handler_input(input, contents),
            },
        }
    }

    pub fn run_loop(mut self, llama: &mut LlamaCtx) -> anyhow::Result<()> {
        // setup terminal
        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let mut main_loop = || -> anyhow::Result<()> {
            let mut prompts = crate::loader_prompt(&self.prompts_path)?;
            let mut output = Output::Normal;
            let mut stream: Option<LlamaModelChatStream<_>> = None;
            let mut start_content = None;

            loop {
                terminal.draw(|f| self.render(&prompts, f))?;

                match output {
                    Output::Exit => break,
                    Output::Chat => {
                        if let Some(c) = prompts.back() {
                            start_content = Some(c.message.clone());
                        }
                        // let option = simple_llama::SimpleOption::Temp(0.9);
                        // let option = simple_llama::SimpleOption::TopP(1.0, 20);
                        // let option = simple_llama::SimpleOption::MirostatV2(3.5, 0.25);
                        let option = simple_llama::SimpleOption::MirostatV2(2.0, 0.25);

                        stream = Some(llama.chat(&prompts, option).unwrap())
                    }
                    Output::Normal => {}
                }

                let input = if let Some(mut stream_) = stream.take() {
                    // interrupt
                    let input = if event::poll(Duration::from_secs(0))? {
                        match event::read()? {
                            Event::Key(input) => {
                                if input.code == KeyCode::Char('c')
                                    && input.modifiers.contains(KeyModifiers::CONTROL)
                                {
                                    continue;
                                } else {
                                    None
                                }
                            }
                            Event::Mouse(input) => Some(Input::Event(Event::Mouse(input))),
                            _ => None,
                        }
                    } else {
                        None
                    };

                    match input {
                        Some(input) => {
                            stream = Some(stream_);
                            input
                        }
                        None => {
                            if let Some(token) = stream_.next() {
                                stream = Some(stream_);
                                Input::Message(Token::Chunk(token))
                            } else {
                                let end: String = stream_.into();
                                let start_content = start_content.take().unwrap_or_default();
                                Input::Message(Token::End(start_content + &end))
                            }
                        }
                    }
                } else {
                    Input::Event(event::read()?)
                };

                output = self.handler_input(input, &mut prompts)?;
            }
            Ok(())
        };

        let r = main_loop();

        // restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;
        r
    }
}
