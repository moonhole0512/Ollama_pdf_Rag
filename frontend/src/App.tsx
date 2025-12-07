import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import {
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  CircularProgress,
  Alert,
  Snackbar,
  CssBaseline,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Button,
  Paper,
  Stack,
  TextField,
  ToggleButtonGroup,
  ToggleButton,
  Divider,
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import { styled } from '@mui/material/styles';
import ChatPanel from './components/ChatPanel';

// --- Interfaces & Constants ---
interface DocumentSource {
  page_content: string;
  metadata: { page: number; source: string; };
}

// Correctly defined interface for Ollama models list
interface OllamaModel {
  name: string;
  modified_at: string;
  size: number;
}

interface ChatMessage {
  type: 'user' | 'ai';
  text: string;
  sources?: DocumentSource[];
}

type ModelProvider = 'ollama' | 'google';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';
const GOOGLE_CHAT_MODELS = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"];
const GOOGLE_EMBEDDING_MODELS = ["models/text-embedding-004", "models/embedding-001"];
const DEFAULT_SYSTEM_PROMPT = '답변은 반드시 한글로만 작성해주세요.';

const DropzoneContainer = styled(Box)(({ theme }) => ({
  flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
  padding: theme.spacing(2), borderWidth: 2, borderRadius: theme.shape.borderRadius,
  borderColor: theme.palette.grey[400], borderStyle: 'dashed', backgroundColor: theme.palette.grey[50],
  color: theme.palette.grey[600], outline: 'none', transition: 'border .24s ease-in-out', cursor: 'pointer',
  '&:hover': { borderColor: theme.palette.primary.main },
}));

// --- App Component ---
function App() {
  // State Management
  const [provider, setProvider] = useState<ModelProvider>('ollama');
  const [googleApiKey, setGoogleApiKey] = useState<string>('');
  const [systemPrompt, setSystemPrompt] = useState<string>(() => localStorage.getItem('systemPrompt') || DEFAULT_SYSTEM_PROMPT);
  const [retrievalK, setRetrievalK] = useState<number>(20);
  
  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([]); // Correctly typed state
  const [chatModel, setChatModel] = useState<string>('');
  const [embeddingModel, setEmbeddingModel] = useState<string>('');
  
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [processingPdf, setProcessingPdf] = useState<boolean>(false);
  const [pdfProcessed, setPdfProcessed] = useState<boolean>(false);
  
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [isGeneratingResponse, setIsGeneratingResponse] = useState<boolean>(false);
  
  const [snackbarOpen, setSnackbarOpen] = useState<boolean>(false);
  const [snackbarMessage, setSnackbarMessage] = useState<string>('');
  const [snackbarSeverity, setSnackbarSeverity] = useState<'success' | 'error' | 'info'>('info');

  const chatHistoryRef = useRef<[string, string][]>([]);

  const showSnackbar = (message: string, severity: 'success' | 'error' | 'info' = 'info') => {
    setSnackbarMessage(message);
    setSnackbarSeverity(severity);
    setSnackbarOpen(true);
  };

  useEffect(() => {
    if (provider === 'ollama') {
      axios.get<{ models: OllamaModel[] }>(`${API_BASE_URL}/api/ollama/models`) // Correct type for axios.get
        .then(response => {
          const models = response.data.models;
          setOllamaModels(models);
          if (models.length > 0) {
            const qwenModel = models.find((m: any) => m.name.startsWith('qwen2.5:14b'));
            const bgeModel = models.find((m: any) => m.name.startsWith('bge-m3'));
            setChatModel(qwenModel ? qwenModel.name : models[0].name);
            setEmbeddingModel(bgeModel ? bgeModel.name : models[0].name);
          }
        })
        .catch(error => {
            console.error("Failed to fetch Ollama models:", error);
            showSnackbar('Failed to fetch Ollama models. Is Ollama running?', 'error');
        });
    } else {
      setChatModel(GOOGLE_CHAT_MODELS[0]);
      setEmbeddingModel(GOOGLE_EMBEDDING_MODELS[0]);
    }
  }, [provider]);

  const handleSystemPromptChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setSystemPrompt(newValue);
    localStorage.setItem('systemPrompt', newValue);
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0 && acceptedFiles[0].type === 'application/pdf') {
      setSelectedFile(acceptedFiles[0]);
      setPdfProcessed(false);
      showSnackbar(`Selected file: ${acceptedFiles[0].name}`, 'info');
    } else {
      showSnackbar('Only PDF files are allowed.', 'error');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  const handlePdfUpload = async () => {
    if (!selectedFile || !embeddingModel) {
      showSnackbar('Please select a PDF and an Embedding Model.', 'error'); return;
    }
    if (provider === 'google' && !googleApiKey) {
      showSnackbar('Please enter your Google API Key.', 'error'); return;
    }

    setProcessingPdf(true); setPdfProcessed(false);
    showSnackbar('Uploading and processing PDF...', 'info');

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('provider', provider);
    formData.append('embedding_model', embeddingModel);
    if (provider === 'google') formData.append('google_api_key', googleApiKey);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/upload`, formData);
      showSnackbar(response.data.message, 'success');
      setPdfProcessed(true);
      setChatMessages([]);
      chatHistoryRef.current = [];
    } catch (error: any) {
      showSnackbar(error.response?.data?.detail || 'Failed to process PDF.', 'error');
    } finally {
      setProcessingPdf(false);
    }
  };

  const handleSendMessage = async (message: string) => {
    if (!pdfProcessed || (provider === 'google' && !googleApiKey)) {
      showSnackbar(pdfProcessed ? 'Please enter your Google API Key.' : 'Please upload and process a PDF first.', 'error'); return;
    }
    
    setIsGeneratingResponse(true);
    setChatMessages(prev => [...prev, { type: 'user', text: message }]);

    try {
      const response = await axios.post<any, { data: { answer: string; source_documents: DocumentSource[] } }>(`${API_BASE_URL}/api/chat`, {
        provider, question: message, chat_history: chatHistoryRef.current,
        chat_model: chatModel, embedding_model: embeddingModel,
        google_api_key: googleApiKey, system_prompt: systemPrompt,
        retrieval_k: retrievalK,
      });
      const { answer, source_documents } = response.data;
      setChatMessages(prev => [...prev, { type: 'ai', text: answer, sources: source_documents }]);
      chatHistoryRef.current = [...chatHistoryRef.current, [message, answer]];
    } catch (error: any) {
      showSnackbar(error.response?.data?.detail || 'Failed to get a response.', 'error');
    } finally {
      setIsGeneratingResponse(false);
    }
  };

  const renderModelSelectors = () => {
    const models = provider === 'ollama' ? ollamaModels.map((m) => m.name) : GOOGLE_CHAT_MODELS;
    const embeddingModels = provider === 'ollama' ? ollamaModels.map((m) => m.name) : GOOGLE_EMBEDDING_MODELS;

    return (
      <Box sx={{ my: 2 }}>
        <FormControl fullWidth sx={{ mb: 2 }}>
          <InputLabel id="chat-model-label">Chat Model</InputLabel>
          <Select labelId="chat-model-label" value={chatModel} label="Chat Model" onChange={(e) => setChatModel(e.target.value)} disabled={models.length === 0 || processingPdf}>
            {models.map((name: string) => (<MenuItem key={name} value={name}>{name}</MenuItem>))}
          </Select>
        </FormControl>
        <FormControl fullWidth>
          <InputLabel id="embedding-model-label">Embedding Model</InputLabel>
          <Select labelId="embedding-model-label" value={embeddingModel} label="Embedding Model" onChange={(e) => setEmbeddingModel(e.target.value)} disabled={embeddingModels.length === 0 || processingPdf}>
            {embeddingModels.map((name: string) => (<MenuItem key={name} value={name}>{name}</MenuItem>))}
          </Select>
        </FormControl>
      </Box>
    );
  };
  
  return (
    <React.Fragment>
      <CssBaseline />
      <AppBar position="sticky"><Toolbar><Typography variant="h6">Ollama & Gemini RAG PDF Chat</Typography></Toolbar></AppBar>
      <Container maxWidth="xl" sx={{ mt: 2, height: 'calc(100vh - 80px)' }}>
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} sx={{ height: '100%' }}>
          <Paper elevation={3} sx={{ width: { xs: '100%', md: '33.33%' }, p: 2, display: 'flex', flexDirection: 'column', overflowY: 'auto' }}>
            <Typography variant="h5" gutterBottom>Settings</Typography>
            <ToggleButtonGroup color="primary" value={provider} exclusive onChange={(_, p) => { if(p) setProvider(p); }} fullWidth sx={{ mb: 2 }}>
              <ToggleButton value="ollama">Ollama</ToggleButton>
              <ToggleButton value="google">Google Gemini</ToggleButton>
            </ToggleButtonGroup>
            <TextField label="System Prompt" value={systemPrompt} onChange={handleSystemPromptChange} fullWidth multiline rows={2} sx={{ mb: 2 }}/>
            {provider === 'google' && <TextField label="Google API Key" type="password" value={googleApiKey} onChange={(e) => setGoogleApiKey(e.target.value)} fullWidth sx={{ mb: 2 }}/>}
            <Divider sx={{ my: 1 }}><Typography variant="overline">Model Selection</Typography></Divider>
            {renderModelSelectors()}
            <Divider sx={{ my: 1 }}><Typography variant="overline">Retrieval</Typography></Divider>
            <TextField label="Chunks to Retrieve (k)" type="number" value={retrievalK} onChange={(e) => setRetrievalK(Number(e.target.value))} fullWidth sx={{ mb: 2 }}/>
            <Divider sx={{ my: 1 }}><Typography variant="overline">PDF Upload</Typography></Divider>
            <DropzoneContainer {...getRootProps()}>
              <input {...getInputProps()} />
              <Typography align="center">{isDragActive ? "Drop PDF here" : "Drag 'n' drop or click"}</Typography>
              {selectedFile && <Typography variant="body2" sx={{ mt: 1 }}>{selectedFile.name}</Typography>}
            </DropzoneContainer>
            <Button variant="contained" onClick={handlePdfUpload} disabled={!selectedFile || processingPdf} sx={{ mt: 2 }}>
              {processingPdf ? <CircularProgress size={24} /> : 'Process PDF'}
            </Button>
            {pdfProcessed && <Alert severity="success" sx={{ mt: 1 }}>PDF is ready.</Alert>}
          </Paper>
          <Paper elevation={3} sx={{ flexGrow: 1, p: 2, display: 'flex', flexDirection: 'column' }}>
            <ChatPanel messages={chatMessages} onSendMessage={handleSendMessage} isGeneratingResponse={isGeneratingResponse} pdfProcessed={pdfProcessed} />
          </Paper>
        </Stack>
      </Container>
      <Snackbar open={snackbarOpen} autoHideDuration={6000} onClose={() => setSnackbarOpen(false)}><Alert severity={snackbarSeverity} sx={{ width: '100%' }}>{snackbarMessage}</Alert></Snackbar>
    </React.Fragment>
  );
}

export default App;
