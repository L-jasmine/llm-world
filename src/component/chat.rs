use std::collections::LinkedList;

use crossterm::event::{Event, KeyCode, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::backend::Backend;
use ratatui::layout::Position;
use ratatui::style::{Color, Style, Stylize};
use ratatui::Terminal;
use ratatui::{
    layout::{Constraint, Layout, Rect},
    text::{Line, Text},
    widgets::{Block, Paragraph},
    Frame,
};
use simple_llama::llm::{Content, Role};
use tui_textarea::TextArea;

use crate::chat::im_channel::Message;
use crate::llm::local_llm::Token;

pub struct MessagesComponent {
    contents: LinkedList<Content>,
    cursor: (u16, u16),
    last_mouse_event: MouseEvent,
    lock_on_bottom: bool,
    pub(super) wait_token: bool,
    area: Rect,
    active: bool,
}

impl MessagesComponent {
    pub fn new(contents: LinkedList<Content>) -> Self {
        Self {
            contents,
            cursor: (0, 0),
            lock_on_bottom: true,
            wait_token: false,
            active: true,
            area: Rect::default(),
            last_mouse_event: MouseEvent {
                row: 0,
                column: 0,
                kind: MouseEventKind::Moved,
                modifiers: KeyModifiers::empty(),
            },
        }
    }

    fn update_active(&mut self, event: MouseEvent) {
        if event.kind == MouseEventKind::Down(MouseButton::Left) {
            self.active = self.area.contains(Position::new(event.column, event.row))
        }
        if event.kind == MouseEventKind::Drag(MouseButton::Left) && self.active {
            self.move_scoll(event);
        }
        self.last_mouse_event = event;
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect)
    where
        Self: Sized,
    {
        self.area = area;
        let mut text = Text::default();
        for content in &self.contents {
            let style = match content.role {
                Role::Assistant => Style::new().bg(Color::Cyan),
                Role::User => Style::new().bg(Color::Yellow),
                Role::Tool => Style::new().bg(Color::Gray),
                _ => Style::new(),
            };
            text.extend([Line::styled(
                format!("{}:", content.role.to_string().to_uppercase()),
                style,
            )]);
            {
                let chars = content.message.chars();
                let max_len = (self.area.width.max(2) - 2) as usize;
                let mut s = String::with_capacity(max_len);
                let mut len = 0;
                for c in chars {
                    s.push(c);
                    if c.is_ascii() {
                        len += 1;
                    } else {
                        len += 2;
                    }
                    if len >= max_len {
                        text.extend(Line::raw(s).style(style));
                        s = String::with_capacity(max_len);
                        len = 0;
                    }
                }
                text.extend(Line::raw(s).style(style));
                // text.extend(Text::raw(&content.message).style(style));
                text.extend([Line::styled(format!("[{max_len},{len}]"), style)]);
            }
        }

        let line_n = text.lines.len();

        let max_line = (area.height - 2 - 1) as usize;
        if line_n > max_line {
            let max_cursor = line_n - max_line;
            if self.cursor.0 >= max_cursor as u16 {
                self.lock_on_bottom = true;
            }

            if self.lock_on_bottom {
                self.cursor.0 = max_cursor as u16;
            }
        } else {
            self.cursor.0 = 0;
        }

        let paragraph = Paragraph::new(text)
            .block(Block::bordered().title(format!("{:?}", self.cursor)).gray())
            .scroll(self.cursor);
        frame.render_widget(paragraph, area);
    }

    pub fn move_scoll(&mut self, event: MouseEvent) {
        let (delta_y, delta_x) = (
            event.row as i16 - self.last_mouse_event.row as i16,
            event.column as i16 - self.last_mouse_event.column as i16,
        );
        if delta_x != 0 {
            self.cursor.1 = (self.cursor.1 as i16 - delta_x).max(0) as u16;
        }
        if delta_y != 0 {
            self.cursor.0 = (self.cursor.0 as i16 - delta_y).max(0) as u16;
            self.lock_on_bottom = false;
        }
    }

    pub fn handler_input(&mut self, input: Input) {
        match input {
            Input::Message(Message::Assistant(Token::Start)) => {
                self.wait_token = true;
            }
            Input::Message(Message::Assistant(Token::Chunk(chunk))) => {
                if let Some(content) = self.contents.back_mut() {
                    content.message.push_str(&chunk);
                }
            }
            Input::Message(Message::Assistant(Token::End(chunk))) => {
                self.wait_token = false;
                if let Some(content) = self.contents.back_mut() {
                    content.message = chunk;
                }
            }

            Input::Event(Event::Mouse(event)) => {
                match event.kind {
                    MouseEventKind::ScrollDown => {
                        if event.modifiers.contains(KeyModifiers::CONTROL) {
                            self.cursor.1 += 6;
                        } else {
                            self.cursor.0 += 3;
                        }
                    }
                    MouseEventKind::ScrollUp => {
                        if event.modifiers.contains(KeyModifiers::CONTROL) {
                            self.cursor.1 = self.cursor.1.max(6) - 6;
                        } else {
                            self.cursor.0 = self.cursor.0.max(3) - 3;
                            self.lock_on_bottom = false;
                        }
                    }
                    _ => {}
                }
                self.update_active(event);
            }
            _ => {}
        }
    }
}

pub struct ChatComponent {
    user_tx: crossbeam::channel::Sender<Message>,
    messages: MessagesComponent,
    input: TextArea<'static>,
    cursor_delta: (i16, i16),
    last_mouse_event: MouseEvent,
    active: bool,
    area: Rect,
    exit_n: u8,
    pub event: String,
    rewrite: bool,
}

#[derive(Debug)]
pub enum Input {
    Event(Event),
    Message(Message),
}

impl ChatComponent {
    pub fn new(
        contents: LinkedList<Content>,
        user_tx: crossbeam::channel::Sender<Message>,
    ) -> Self {
        Self {
            messages: MessagesComponent::new(contents),
            input: Self::new_textarea(),
            exit_n: 0,
            event: String::new(),
            user_tx,
            rewrite: false,
            cursor_delta: (0, 0),
            last_mouse_event: MouseEvent {
                kind: MouseEventKind::Moved,
                column: 0,
                row: 0,
                modifiers: KeyModifiers::empty(),
            },
            active: false,
            area: Rect::default(),
        }
    }

    fn update_active(&mut self, event: MouseEvent) {
        if event.kind == MouseEventKind::Down(MouseButton::Left) {
            self.active = self.area.contains(Position::new(event.column, event.row))
        }
        if event.kind == MouseEventKind::Drag(MouseButton::Left) && self.active {
            self.move_scoll(event)
        } else {
            self.cursor_delta = (0, 0);
        }
        self.last_mouse_event = event;
    }

    pub fn move_scoll(&mut self, event: MouseEvent) {
        let (delta_y, delta_x) = (
            event.row as i16 - self.last_mouse_event.row as i16,
            event.column as i16 - self.last_mouse_event.column as i16,
        );

        self.cursor_delta = (delta_y, delta_x);
    }

    pub fn render(&mut self, frame: &mut Frame, area: Rect)
    where
        Self: Sized,
    {
        let vertical = Layout::vertical([Constraint::Min(5), Constraint::Max(10)]);
        let [messages_area, input_area] = vertical.areas(area);

        self.area = input_area;

        self.messages.render(frame, messages_area);
        if self.messages.wait_token {
            self.input
                .set_block(Block::bordered().title("Input").yellow())
        } else {
            self.input
                .set_block(Block::bordered().title("Input").gray())
        }
        self.input
            .scroll((-self.cursor_delta.0, -self.cursor_delta.1));
        frame.render_widget(self.input.widget(), input_area);
    }

    fn new_textarea() -> TextArea<'static> {
        TextArea::default()
    }

    fn pop_last_assaistant(&mut self) {
        if let Some(content) = self.messages.contents.back_mut() {
            if content.role == Role::Assistant {
                self.input.select_all();
                self.input.cut();
                self.input.insert_str(&content.message);
                content.message.clear();

                self.rewrite = true;
            }
        }
    }

    fn submit_message(&mut self) {
        let mut new_textarea = Self::new_textarea();
        std::mem::swap(&mut self.input, &mut new_textarea);
        let lines = new_textarea.into_lines();
        let message = lines.join("\n");

        if self.rewrite {
            let assistant = self.messages.contents.back_mut().unwrap();
            assistant.message = message;
            let assistant = assistant.clone();

            self.user_tx.send(Message::Generate(assistant)).unwrap();
            self.rewrite = false;
        } else {
            let user = Content {
                role: Role::User,
                message,
            };
            self.messages.contents.push_back(user.clone());
            self.messages.contents.push_back(Content {
                role: Role::Assistant,
                message: String::new(),
            });

            self.user_tx.send(Message::GenerateByUser(user)).unwrap();
        }
        self.messages.lock_on_bottom = true;
    }

    pub fn handler_input<B: Backend>(&mut self, terminal: &mut Terminal<B>, input: Input) -> bool {
        self.event = format!("{:?}", input);
        let is_event = matches!(&input, Input::Event(..));

        match input {
            Input::Event(Event::Key(input)) if input.code == KeyCode::F(5) => {
                let _ = terminal.clear();
            }
            Input::Event(Event::Key(input))
                if (input.code == KeyCode::Char('j')
                    && input.modifiers.contains(KeyModifiers::CONTROL)) =>
            {
                if !self.messages.wait_token {
                    self.submit_message();
                }
            }
            Input::Event(Event::Key(input))
                if (input.code == KeyCode::Char('r')
                    && input.modifiers.contains(KeyModifiers::CONTROL)) =>
            {
                if !self.messages.wait_token {
                    self.pop_last_assaistant();
                }
            }
            Input::Event(Event::Key(input)) if input.code == KeyCode::Esc => {
                self.exit_n += 2;
                return self.exit_n < 3;
            }
            Input::Event(Event::Key(input)) => {
                self.input.input(input);
            }
            Input::Event(Event::Mouse(event)) => {
                self.update_active(event);
                if !self.active {
                    self.messages
                        .handler_input(Input::Event(Event::Mouse(event)));
                }
            }
            input => {
                self.messages.handler_input(input);
            }
        }

        if is_event {
            self.exit_n = self.exit_n.max(1) - 1;
        }
        true
    }
}
