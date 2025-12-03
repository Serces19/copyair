import React, { useState } from 'react';
import { Upload, Play, Image as ImageIcon, Film } from 'lucide-react';
import { uploadFile, runScript } from '../api/client';

const Inference = () => {
    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [running, setRunning] = useState(false);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };

    const handleUpload = async () => {
        if (!file) return;
        setUploading(true);
        try {
            await uploadFile('01_raw/input', file);
            alert('File uploaded successfully!');
        } catch (error) {
            console.error(error);
            alert('Upload failed');
        } finally {
            setUploading(false);
        }
    };

    const handleRun = async () => {
        setRunning(true);
        try {
            await runScript('predict');
            alert('Inference started!');
        } catch (error) {
            console.error(error);
            alert('Failed to start inference');
        } finally {
            setRunning(false);
        }
    };

    const handleBatchInference = async () => {
        setRunning(true);
        try {
            await runScript('batch_inference');
            alert('Batch Inference started!');
        } catch (error) {
            console.error(error);
            alert('Failed to start batch inference');
        } finally {
            setRunning(false);
        }
    };

    return (
        <div>
            <h2 className="text-3xl font-bold mb-6">Inference</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Upload Section */}
                <div className="card">
                    <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                        <Upload size={20} className="text-blue-400" />
                        Upload Input
                    </h3>

                    <div className="border-2 border-dashed border-white/10 rounded-lg p-8 text-center hover:border-blue-500/50 transition-colors">
                        <input
                            type="file"
                            id="file-upload"
                            className="hidden"
                            onChange={handleFileChange}
                        />
                        <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center gap-2">
                            {file ? (
                                <div className="text-green-400 font-medium">{file.name}</div>
                            ) : (
                                <>
                                    <ImageIcon size={40} className="text-gray-500" />
                                    <span className="text-gray-400">Click to upload image or video</span>
                                </>
                            )}
                        </label>
                    </div>

                    <button
                        onClick={handleUpload}
                        disabled={!file || uploading}
                        className="btn btn-secondary w-full mt-4"
                    >
                        {uploading ? 'Uploading...' : 'Upload to Input Folder'}
                    </button>
                </div>

                {/* Run Section */}
                <div className="card">
                    <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                        <Play size={20} className="text-purple-400" />
                        Run Inference
                    </h3>

                    <p className="text-gray-400 mb-6">
                        Run the model on the uploaded images/videos in the input folder.
                        Results will be saved to <code>output_inference</code>.
                    </p>

                    <div className="flex flex-col gap-3">
                        <button
                            onClick={handleRun}
                            disabled={running}
                            className="btn btn-primary w-full"
                        >
                            {running ? 'Running...' : 'Start Single Inference'}
                        </button>
                        <button
                            onClick={handleBatchInference}
                            disabled={running}
                            className="btn btn-secondary w-full"
                        >
                            <Film size={18} />
                            {running ? 'Running...' : 'Run Batch Inference'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Inference;
