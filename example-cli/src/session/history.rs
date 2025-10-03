use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub timestamp: i64,
}

impl Message {
    pub fn new_user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    pub fn new_assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    pub fn new_system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            timestamp: chrono::Utc::now().timestamp(),
        }
    }

    pub fn is_user(&self) -> bool {
        self.role == "user"
    }

    pub fn is_assistant(&self) -> bool {
        self.role == "assistant"
    }

    pub fn is_system(&self) -> bool {
        self.role == "system"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionHistory {
    messages: Vec<Message>,
}

impl SessionHistory {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    pub fn add_user_message(&mut self, content: &str) {
        self.messages.push(Message::new_user(content));
    }

    pub fn add_assistant_message(&mut self, content: &str) {
        self.messages.push(Message::new_assistant(content));
    }

    pub fn add_system_message(&mut self, content: &str) {
        self.messages.push(Message::new_system(content));
    }

    pub fn clear(&mut self) {
        self.messages.clear();
    }

    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn last_message(&self) -> Option<&Message> {
        self.messages.last()
    }

    pub fn user_messages(&self) -> impl Iterator<Item = &Message> {
        self.messages.iter().filter(|m| m.is_user())
    }

    pub fn assistant_messages(&self) -> impl Iterator<Item = &Message> {
        self.messages.iter().filter(|m| m.is_assistant())
    }
}

impl Default for SessionHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let user_msg = Message::new_user("Hello");
        assert!(user_msg.is_user());
        assert!(!user_msg.is_assistant());
        assert_eq!(user_msg.content, "Hello");

        let assistant_msg = Message::new_assistant("Hi");
        assert!(assistant_msg.is_assistant());
        assert!(!assistant_msg.is_user());
        assert_eq!(assistant_msg.content, "Hi");

        let system_msg = Message::new_system("System");
        assert!(system_msg.is_system());
        assert!(!system_msg.is_user());
    }

    #[test]
    fn test_session_history_new() {
        let history = SessionHistory::new();
        assert_eq!(history.message_count(), 0);
    }

    #[test]
    fn test_add_messages() {
        let mut history = SessionHistory::new();

        history.add_user_message("Hello");
        assert_eq!(history.message_count(), 1);

        history.add_assistant_message("Hi there!");
        assert_eq!(history.message_count(), 2);

        history.add_system_message("System message");
        assert_eq!(history.message_count(), 3);
    }

    #[test]
    fn test_clear() {
        let mut history = SessionHistory::new();
        history.add_user_message("Test");
        assert_eq!(history.message_count(), 1);

        history.clear();
        assert_eq!(history.message_count(), 0);
    }

    #[test]
    fn test_last_message() {
        let mut history = SessionHistory::new();
        assert!(history.last_message().is_none());

        history.add_user_message("First");
        history.add_assistant_message("Second");

        let last = history.last_message().unwrap();
        assert_eq!(last.content, "Second");
        assert!(last.is_assistant());
    }

    #[test]
    fn test_filter_messages() {
        let mut history = SessionHistory::new();
        history.add_user_message("User1");
        history.add_assistant_message("Assistant1");
        history.add_user_message("User2");

        let user_count = history.user_messages().count();
        assert_eq!(user_count, 2);

        let assistant_count = history.assistant_messages().count();
        assert_eq!(assistant_count, 1);
    }

    #[test]
    fn test_serialization() {
        let mut history = SessionHistory::new();
        history.add_user_message("Test");
        history.add_assistant_message("Response");

        let json = serde_json::to_string(&history).unwrap();
        let deserialized: SessionHistory = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.message_count(), 2);
    }
}
