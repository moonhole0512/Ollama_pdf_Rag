import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

interface ChatRequest {
  question: string;
  chat_history?: [string, string][];
  chat_model: string;
  embedding_model: string;
}

interface ChatResponse {
  answer: string;
}

interface OllamaModel {
  model: string;
  name?: string;
  modified_at: string;
  size: number;
}

interface UploadResponse {
  status: string;
  filename: string;
  message: string;
}

export const api = {
  getOllamaModels: async (): Promise<OllamaModel[]> => {
    const response = await axios.get(`${API_BASE_URL}/api/ollama/models`);
    return response.data.models.map((m: any) => ({
      model: m.model,
      name: m.model.split(':')[0],
      modified_at: m.modified_at,
      size: m.size,
    }));
  },

  uploadPdf: async (file: File, embeddingModel: string): Promise<UploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('embedding_model', embeddingModel);
    const response = await axios.post(`${API_BASE_URL}/api/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  chat: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await axios.post(`${API_BASE_URL}/api/chat`, request);
    return response.data;
  },
};
