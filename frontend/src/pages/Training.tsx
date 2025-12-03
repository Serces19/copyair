import React, { useState, useEffect } from 'react';
import { Play, Square, Terminal } from 'lucide-react';
import { runScript, getLogs, getStatus } from '../api/client';

const Training = () => {
    const [status, setStatus] = useState('stopped');
    const [logs, setLogs] = useState<string[]>([]);

    const handleStart = async () => {
        try {
            await runScript('train');
            setStatus('running');
        } catch (error) {
            console.error(error);
        }
    };

    const handleGridSearch = async () => {
        try {
            await runScript('hyperparam_search');
            setStatus('running_grid');
        } catch (error) {
            console.error(error);
        }
    };

    useEffect(() => {
        const interval = setInterval(async () => {
            if (status.startsWith('running')) {
                const scriptType = status === 'running' ? 'train' : 'hyperparam_search';
                const logData = await getLogs(scriptType);
                setLogs(logData.logs);

                const statusData = await getStatus(scriptType);
                if (!statusData.status.startsWith('running')) {
                    setStatus(statusData.status);
                }
            }
        }, 1000);
        return () => clearInterval(interval);
    }, [status]);

    return (
        <div>
            <div className="flex justify-between items-center mb-6">
                <h2 className="text-3xl font-bold">Training</h2>
                <div className="flex gap-4">
                    <button
                        onClick={handleGridSearch}
                        disabled={status.startsWith('running')}
                        className={`btn ${status === 'running_grid' ? 'btn-secondary' : 'btn-secondary'}`}
                    >
                        {status === 'running_grid' ? <Square size={18} /> : <Terminal size={18} />}
                        {status === 'running_grid' ? 'Grid Search Running...' : 'Run Grid Search'}
                    </button>
                    <button
                        onClick={handleStart}
                        disabled={status.startsWith('running')}
                        className={`btn ${status === 'running' ? 'btn-secondary' : 'btn-primary'}`}
                    >
                        {status === 'running' ? <Square size={18} /> : <Play size={18} />}
                        {status === 'running' ? 'Running...' : 'Start Training'}
                    </button>
                </div>
            </div>

            <div className="card bg-black/50 font-mono text-sm h-[600px] overflow-auto p-4 border-white/10">
                {logs.length === 0 ? (
                    <div className="text-gray-500 flex items-center justify-center h-full gap-2">
                        <Terminal size={20} />
                        <span>Ready to start training...</span>
                    </div>
                ) : (
                    logs.map((log, i) => (
                        <div key={i} className="whitespace-pre-wrap text-gray-300">{log}</div>
                    ))
                )}
            </div>
        </div>
    );
};

export default Training;
