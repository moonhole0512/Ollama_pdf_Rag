import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

// --- Interfaces ---
export interface ChatRequest {
  provider: string;
  question: string;
  chat_model: string;
  embedding_model: string;
  router_model?: string;
  google_api_key?: string;
  system_prompt?: string;
  retrieval_k?: number;
  chat_history?: [string, string][];
}

export interface SetActiveDocsRequest {
  document_names: string[];
  provider: string;
  embedding_model: string;
  google_api_key?: string;
}

export interface ChatResponse {
  answer: string;
  source_documents: any[]; 
}

export interface OllamaModel {
  name: string;
  modified_at: string;
  size: number;
}

export interface UploadResponse {
  status: string;
  filename: string;
  message: string;
}

export interface ProgressData {
  current: number;
  total: number;
  status: string;
  message: string;
}

// --- API Service ---
export const api = {
  getOllamaModels: async (): Promise<OllamaModel[]> => {
    const response = await axios.get<{ models: any[] }>(`${API_BASE_URL}/api/ollama/models`);
    return response.data.models;
  },
  
  getDocuments: async (): Promise<string[]> => {
    const response = await axios.get<string[]>(`${API_BASE_URL}/api/documents`);
    return response.data;
  },

  uploadPdf: async (file: File, provider: string, embeddingModel: string, googleApiKey?: string): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('provider', provider);
    formData.append('embedding_model', embeddingModel);
    if (provider === 'google' && googleApiKey) {
      formData.append('google_api_key', googleApiKey);
    }
    const response = await axios.post<UploadResponse>(`${API_BASE_URL}/api/upload`, formData);
    return response.data;
  },

  getProgress: async (): Promise<ProgressData> => {
    const response = await axios.get<ProgressData>(`${API_BASE_URL}/api/progress`);
    return response.data;
  },
  
  setActiveDocuments: async (req: SetActiveDocsRequest): Promise<{message: string}> => {
    const response = await axios.post(`${API_BASE_URL}/api/set-active-documents`, req);
    return response.data;
  },

  chat: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await axios.post<ChatResponse>(`${API_BASE_URL}/api/chat`, request);
    return response.data;
  },

  deleteDocument: async (docName: string): Promise<{status: string, message: string}> => {
    const response = await axios.delete(`${API_BASE_URL}/api/documents`, {
      data: { doc_name: docName },
    });
    return response.data;
  },
};