import React, { useEffect, useRef } from 'react';
import {
  Box,
  TextField,
  Button,
  CircularProgress,
  List,
  ListItemText,
  Paper,
  Typography,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import { styled } from '@mui/material/styles';
import SourceAccordion from './SourceAccordion'; // Import the new component

// --- Interfaces ---
interface DocumentSource {
  page_content: string;
  metadata: {
    page: number;
    source: string;
  };
}

interface ChatMessage {
  type: 'user' | 'ai';
  text: string;
  sources?: DocumentSource[];
}

interface ChatPanelProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
  isGeneratingResponse: boolean;
  pdfProcessed: boolean;
}

// --- Styled Components ---
const MessageContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'flex-start',
  marginBottom: theme.spacing(2),
  '&.user': {
    alignItems: 'flex-end',
  },
}));

const MessageBubble = styled(Paper)(({ theme }) => ({
  maxWidth: '80%',
  padding: theme.spacing(1.5),
  borderRadius: theme.shape.borderRadius,
  backgroundColor: theme.palette.grey[100],
  color: theme.palette.text.primary,
  '&.user': {
    backgroundColor: theme.palette.primary.main,
    color: theme.palette.primary.contrastText,
  },
}));

const ChatInputContainer = styled(Box)({
  display: 'flex',
  paddingTop: '16px',
  borderTop: '1px solid #e0e0e0',
});

// --- Component ---
const ChatPanel: React.FC<ChatPanelProps> = ({
  messages,
  onSendMessage,
  isGeneratingResponse,
  pdfProcessed,
}) => {
  const [inputMessage, setInputMessage] = React.useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSendClick = () => {
    if (inputMessage.trim()) {
      onSendMessage(inputMessage);
      setInputMessage('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendClick();
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <Typography variant="h5" gutterBottom>Chat</Typography>
      <Box sx={{ flexGrow: 1, overflowY: 'auto', mb: 2, p: 1 }}>
        <List>
          {messages.length === 0 && (
            <Typography variant="body2" color="textSecondary" align="center" sx={{ mt: 4 }}>
              {pdfProcessed ? "Ask a question about your PDF." : "Upload and process a PDF to start chatting."}
            </Typography>
          )}
          {messages.map((msg, index) => (
            <MessageContainer key={index} className={msg.type}>
              <MessageBubble className={msg.type}>
                <ListItemText
                  primary={msg.text}
                  sx={{ my: 0, whiteSpace: 'pre-wrap' }}
                  primaryTypographyProps={{ color: msg.type === 'user' ? 'inherit' : 'textPrimary' }}
                />
              </MessageBubble>
              {msg.type === 'ai' && msg.sources && <SourceAccordion sources={msg.sources} />}
            </MessageContainer>
          ))}
          {isGeneratingResponse && (
            <MessageContainer>
              <MessageBubble>
                <CircularProgress size={20} sx={{ verticalAlign: 'middle', mr: 1 }} />
                <Typography variant="body2" component="span" sx={{ verticalAlign: 'middle' }}>
                  AI is thinking...
                </Typography>
              </MessageBubble>
            </MessageContainer>
          )}
          <div ref={messagesEndRef} />
        </List>
      </Box>
      <ChatInputContainer>
        <TextField
          fullWidth
          multiline
          maxRows={5}
          variant="outlined"
          placeholder={pdfProcessed ? "Type your message..." : "Please process a PDF first"}
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={!pdfProcessed || isGeneratingResponse}
          sx={{ mr: 1 }}
        />
        <Button
          variant="contained"
          onClick={handleSendClick}
          disabled={!pdfProcessed || isGeneratingResponse || inputMessage.trim() === ''}
          endIcon={<SendIcon />}
        >
          Send
        </Button>
      </ChatInputContainer>
    </Box>
  );
};

export default ChatPanel;