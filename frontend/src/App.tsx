import React, { useState, useEffect, useCallback, useRef } from 'react';
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
  Checkbox,
  FormGroup,
  FormControlLabel,
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import { styled } from '@mui/material/styles';
import ChatPanel from './components/ChatPanel';
import { api, type SetActiveDocsRequest } from './api'; // Import api service and new interface

// --- Interfaces & Constants ---
interface DocumentSource {
  page_content: string;
  metadata: { page: number; source: string; };
}

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
  const [retrievalK, setRetrievalK] = useState<number>(5); // Default to 5 for Ensemble Retriever

  const [ollamaModels, setOllamaModels] = useState<OllamaModel[]>([]);
  const [chatModel, setChatModel] = useState<string>('');
  const [embeddingModel, setEmbeddingModel] = useState<string>('');
  
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [processingPdf, setProcessingPdf] = useState<boolean>(false);
  
  // New state for document management
  const [allDocuments, setAllDocuments] = useState<string[]>([]);
  const [selectedDocuments, setSelectedDocuments] = useState<Set<string>>(new Set());
  const [isSessionActive, setIsSessionActive] = useState<boolean>(false);
  const [isActivatingSession, setIsActivatingSession] = useState<boolean>(false);

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

  // --- API Calls ---
  const fetchOllamaModels = useCallback(async () => {
    if (provider === 'ollama') {
      try {
        const models = await api.getOllamaModels();
        setOllamaModels(models);
        if (models.length > 0) {
          const qwenModel = models.find((m: any) => m.name.startsWith('qwen2.5:14b'));
          const bgeModel = models.find((m: any) => m.name.startsWith('bge-m3'));
          setChatModel(qwenModel ? qwenModel.name : models[0].name);
          setEmbeddingModel(bgeModel ? bgeModel.name : models[0].name);
        }
      } catch (error) {
          console.error("Failed to fetch Ollama models:", error);
          showSnackbar('Failed to fetch Ollama models. Is Ollama running?', 'error');
      }
    } else {
      setChatModel(GOOGLE_CHAT_MODELS[0]);
      setEmbeddingModel(GOOGLE_EMBEDDING_MODELS[0]);
    }
  }, [provider]);

  const fetchDocuments = useCallback(async () => {
    try {
      const docs = await api.getDocuments();
      setAllDocuments(docs);
    } catch (error) {
      console.error("Failed to fetch documents:", error);
      showSnackbar('Failed to fetch processed documents.', 'error');
    }
  }, []);

  // --- Effects ---
  useEffect(() => {
    fetchOllamaModels();
  }, [fetchOllamaModels]);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  useEffect(() => {
    localStorage.setItem('systemPrompt', systemPrompt);
  }, [systemPrompt]);

  // --- Handlers ---
  const handleSystemPromptChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSystemPrompt(e.target.value);
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0 && acceptedFiles[0].type === 'application/pdf') {
      setSelectedFile(acceptedFiles[0]);
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

    setProcessingPdf(true);
    setIsSessionActive(false); // Deactivate session if a new PDF is uploaded
    showSnackbar('Uploading and processing PDF...', 'info');

    try {
      const response = await api.uploadPdf(selectedFile, provider, embeddingModel, googleApiKey || undefined);
      showSnackbar(response.message, 'success');
      setSelectedFile(null); // Clear selected file after upload
      fetchDocuments(); // Refresh document list
    } catch (error: any) {
      showSnackbar(error.response?.data?.detail || 'Failed to process PDF.', 'error');
    } finally {
      setProcessingPdf(false);
    }
  };

  const handleDocumentSelection = (docName: string, isChecked: boolean) => {
    setSelectedDocuments(prev => {
      const newSelection = new Set(prev);
      if (isChecked) {
        newSelection.add(docName);
      } else {
        newSelection.delete(docName);
      }
      return newSelection;
    });
  };

  const handleActivateSession = async () => {
    if (selectedDocuments.size === 0) {
      showSnackbar('Please select at least one document to activate RAG session.', 'error');
      return;
    }
    if (!embeddingModel) {
      showSnackbar('Please select an Embedding Model.', 'error');
      return;
    }
    if (provider === 'google' && !googleApiKey) {
      showSnackbar('Please enter your Google API Key.', 'error');
      return;
    }

    setIsActivatingSession(true);
    setIsSessionActive(false); // Deactivate current session until new one is ready
    setChatMessages([]); // Clear chat history
    chatHistoryRef.current = [];

    showSnackbar('Activating RAG session...', 'info');

    try {
      const reqPayload: SetActiveDocsRequest = {
        document_names: Array.from(selectedDocuments),
        provider,
        embedding_model: embeddingModel,
        google_api_key: googleApiKey || undefined,
      };
      const response = await api.setActiveDocuments(reqPayload);
      showSnackbar(response.message, 'success');
      setIsSessionActive(true);
    } catch (error: any) {
      showSnackbar(error.response?.data?.detail || 'Failed to activate RAG session.', 'error');
    } finally {
      setIsActivatingSession(false);
    }
  };

  const handleSendMessage = async (message: string) => {
    if (!isSessionActive) {
      showSnackbar('Please activate a RAG session by selecting documents first.', 'error'); return;
    }
    if (provider === 'google' && !googleApiKey) {
      showSnackbar('Please enter your Google API Key.', 'error'); return;
    }
    
    setIsGeneratingResponse(true);
    setChatMessages(prev => [...prev, { type: 'user', text: message }]);

    try {
      const response = await api.chat({
        provider, question: message, chat_history: chatHistoryRef.current,
        chat_model: chatModel, embedding_model: embeddingModel,
        google_api_key: googleApiKey, system_prompt: systemPrompt,
        retrieval_k: retrievalK,
      });
      const { answer, source_documents } = response; // api.chat now returns the full response data
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
          <Select labelId="chat-model-label" value={chatModel} label="Chat Model" onChange={(e) => setChatModel(e.target.value)} disabled={models.length === 0 || processingPdf || isActivatingSession}>
            {models.map((name: string) => (<MenuItem key={name} value={name}>{name}</MenuItem>))}
          </Select>
        </FormControl>
        <FormControl fullWidth>
          <InputLabel id="embedding-model-label">Embedding Model</InputLabel>
          <Select labelId="embedding-model-label" value={embeddingModel} label="Embedding Model" onChange={(e) => setEmbeddingModel(e.target.value)} disabled={embeddingModels.length === 0 || processingPdf || isActivatingSession}>
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

            <Divider sx={{ my: 2 }}><Typography variant="overline">Select Documents for RAG</Typography></Divider>
            {allDocuments.length === 0 ? (
              <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>No processed documents found. Upload a PDF first.</Typography>
            ) : (
              <FormGroup sx={{ mb: 2 }}>
                {allDocuments.map(docName => (
                  <FormControlLabel
                    key={docName}
                    control={
                      <Checkbox
                        checked={selectedDocuments.has(docName)}
                        onChange={(e) => handleDocumentSelection(docName, e.target.checked)}
                        name={docName}
                      />
                    }
                    label={docName}
                  />
                ))}
              </FormGroup>
            )}
            <Button variant="contained" onClick={handleActivateSession} disabled={selectedDocuments.size === 0 || isActivatingSession || !embeddingModel} sx={{ mb: 2 }}>
              {isActivatingSession ? <CircularProgress size={24} /> : 'Start RAG Session'}
            </Button>
            {isSessionActive && <Alert severity="success" sx={{ mt: 1 }}>RAG Session Active!</Alert>}

          </Paper>
          <Paper elevation={3} sx={{ flexGrow: 1, p: 2, display: 'flex', flexDirection: 'column' }}>
            <ChatPanel messages={chatMessages} onSendMessage={handleSendMessage} isGeneratingResponse={isGeneratingResponse} isSessionActive={isSessionActive} />
          </Paper>
        </Stack>
      </Container>
      <Snackbar open={snackbarOpen} autoHideDuration={6000} onClose={() => setSnackbarOpen(false)}><Alert severity={snackbarSeverity} sx={{ width: '100%' }}>{snackbarMessage}</Alert></Snackbar>
    </React.Fragment>
  );
}

export default App;
