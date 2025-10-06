use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame, Terminal,
};
use std::io;

use super::commands::Command;
use crate::model::InferenceEngine;
use crate::session::SessionManager;

pub struct TuiApp {
    session: SessionManager,
    engine: InferenceEngine,
    input: String,
    messages: Vec<String>,
    use_chat_template: bool,
    should_quit: bool,
}

impl TuiApp {
    pub fn new(
        session: SessionManager,
        engine: InferenceEngine,
        use_chat_template: bool,
    ) -> Self {
        Self {
            session,
            engine,
            input: String::new(),
            messages: vec![
                "Welcome to RusTorch CLI".to_string(),
                "Type your message and press Enter".to_string(),
                "Press Shift+TAB to toggle chat template mode".to_string(),
                "Type /help for commands, /exit to quit".to_string(),
            ],
            use_chat_template,
            should_quit: false,
        }
    }

    pub fn run(&mut self) -> Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Main loop
        while !self.should_quit {
            terminal.draw(|f| self.ui(f))?;
            self.handle_events()?;
        }

        // Restore terminal
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        Ok(())
    }

    fn ui(&self, f: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(3),      // Messages area
                Constraint::Length(3),   // Input area
                Constraint::Length(1),   // Status bar
            ])
            .split(f.area());

        // Messages area
        self.render_messages(f, chunks[0]);

        // Input area
        self.render_input(f, chunks[1]);

        // Status bar
        self.render_status_bar(f, chunks[2]);
    }

    fn render_messages(&self, f: &mut Frame, area: Rect) {
        let messages_text: Vec<Line> = self
            .messages
            .iter()
            .map(|m| Line::from(m.clone()))
            .collect();

        let messages = Paragraph::new(messages_text)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Chat History"),
            )
            .style(Style::default().fg(Color::White));

        f.render_widget(messages, area);
    }

    fn render_input(&self, f: &mut Frame, area: Rect) {
        let template_indicator = if self.use_chat_template { "📋 " } else { "" };
        let input_text = format!("{}You> {}", template_indicator, self.input);

        let input = Paragraph::new(input_text)
            .block(Block::default().borders(Borders::ALL).title("Input"))
            .style(Style::default().fg(Color::Yellow));

        f.render_widget(input, area);
    }

    fn render_status_bar(&self, f: &mut Frame, area: Rect) {
        let template_status = if self.use_chat_template {
            Span::styled(
                "Template: ON",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Span::styled(
                "Template: OFF",
                Style::default()
                    .fg(Color::Red)
                    .add_modifier(Modifier::BOLD),
            )
        };

        let status_text = Line::from(vec![
            template_status,
            Span::raw(" | "),
            Span::styled(
                "(Shift+TAB to toggle)",
                Style::default().fg(Color::DarkGray),
            ),
            Span::raw(" | "),
            Span::styled(
                format!("Backend: {}", self.session.backend_name()),
                Style::default().fg(Color::Cyan),
            ),
        ]);

        let status_bar = Paragraph::new(status_text)
            .style(Style::default().bg(Color::Black).fg(Color::White));

        f.render_widget(status_bar, area);
    }

    fn handle_events(&mut self) -> Result<()> {
        if event::poll(std::time::Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                self.handle_key_event(key)?;
            }
        }
        Ok(())
    }

    fn handle_key_event(&mut self, key: KeyEvent) -> Result<()> {
        match (key.code, key.modifiers) {
            // Shift+TAB key to toggle template
            (KeyCode::BackTab, _) => {
                self.use_chat_template = !self.use_chat_template;
                let status = if self.use_chat_template { "ON" } else { "OFF" };
                self.messages.push(format!("✓ Template toggled: {}", status));
            }

            // Ctrl+C to quit
            (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }

            // Ctrl+D to quit
            (KeyCode::Char('d'), KeyModifiers::CONTROL) => {
                self.should_quit = true;
            }

            // Enter to submit
            (KeyCode::Enter, _) => {
                if !self.input.is_empty() {
                    self.handle_input()?;
                }
            }

            // Backspace
            (KeyCode::Backspace, _) => {
                self.input.pop();
            }

            // Character input
            (KeyCode::Char(c), _) => {
                self.input.push(c);
            }

            _ => {}
        }
        Ok(())
    }

    fn handle_input(&mut self) -> Result<()> {
        let input = self.input.clone();
        self.input.clear();

        // Add user message to history
        self.messages.push(format!("You> {}", input));

        // Handle commands
        if input.starts_with('/') {
            match Command::parse(&input) {
                Ok(Command::Exit) => {
                    self.should_quit = true;
                }
                Ok(Command::Help) => {
                    self.messages.push(Command::help_text().to_string());
                }
                Ok(Command::TemplateToggle) => {
                    self.use_chat_template = !self.use_chat_template;
                    let status = if self.use_chat_template { "ON" } else { "OFF" };
                    self.messages.push(format!("✓ Template: {}", status));
                }
                Ok(_) => {
                    self.messages
                        .push("Command not implemented in TUI mode yet".to_string());
                }
                Err(e) => {
                    self.messages.push(format!("Error: {}", e));
                }
            }
        } else {
            // Generate response
            self.session.add_user_message(&input);
            match self.engine.generate(&input) {
                Ok(response) => {
                    self.messages.push(format!("Assistant> {}", response));
                    self.session.add_assistant_message(&response);
                }
                Err(e) => {
                    self.messages.push(format!("Error: {}", e));
                }
            }
        }

        Ok(())
    }
}
