import axios from 'axios';

const API_URL = 'http://localhost:8000/api';

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getConfig = async () => {
  const response = await api.get('/config');
  return response.data;
};

export const updateConfig = async (config: any) => {
  const response = await api.post('/config', { config });
  return response.data;
};

export const runScript = async (scriptName: string, args: string[] = []) => {
  const response = await api.post(`/run/${scriptName}`, { script_name: scriptName, args });
  return response.data;
};

export const getStatus = async (scriptName: string) => {
  const response = await api.get(`/status/${scriptName}`);
  return response.data;
};

export const getLogs = async (scriptName: string) => {
  const response = await api.get(`/logs/${scriptName}`);
  return response.data;
};

export const uploadFile = async (folder: string, file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post(`/upload/${folder}`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};
