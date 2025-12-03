import React, { useEffect, useState } from 'react';
import { getConfig, updateConfig } from '../api/client';

const Settings = () => {
    const [config, setConfig] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadConfig();
    }, []);

    const loadConfig = async () => {
        try {
            const data = await getConfig();
            setConfig(data);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        try {
            await updateConfig(config);
            alert('Configuration saved!');
        } catch (error) {
            console.error(error);
            alert('Failed to save configuration');
        }
    };

    if (loading) return <div>Loading...</div>;

    return (
        <div>
            <div className="flex justify-between items-center mb-6">
                <h2 className="text-3xl font-bold">Settings</h2>
                <button onClick={handleSave} className="btn btn-primary">
                    Save Changes
                </button>
            </div>

            <div className="grid gap-6">
                {/* Model Settings */}
                <div className="card">
                    <h3 className="text-xl font-bold mb-4 text-blue-400">Model Configuration</h3>
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Architecture</label>
                            <select
                                className="input bg-black/20"
                                value={config.model.architecture}
                                onChange={(e) => setConfig({ ...config, model: { ...config.model, architecture: e.target.value } })}
                            >
                                <option value="unet">UNet</option>
                                <option value="nafnet">NAFNet</option>
                                <option value="convnext">ConvNeXt</option>
                                <option value="mambair">MambaIR</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Size</label>
                            <select
                                className="input bg-black/20"
                                value={config.model.size}
                                onChange={(e) => setConfig({ ...config, model: { ...config.model, size: e.target.value } })}
                            >
                                <option value="nano">Nano</option>
                                <option value="tiny">Tiny</option>
                                <option value="base">Base</option>
                            </select>
                        </div>
                    </div>
                </div>

                {/* Training Settings */}
                <div className="card">
                    <h3 className="text-xl font-bold mb-4 text-purple-400">Training Parameters</h3>
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Epochs</label>
                            <input
                                type="number"
                                className="input"
                                value={config.training.epochs}
                                onChange={(e) => setConfig({ ...config, training: { ...config.training, epochs: parseInt(e.target.value) } })}
                            />
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Batch Size</label>
                            <input
                                type="number"
                                className="input"
                                value={config.training.batch_size}
                                onChange={(e) => setConfig({ ...config, training: { ...config.training, batch_size: parseInt(e.target.value) } })}
                            />
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Learning Rate</label>
                            <input
                                type="text"
                                className="input"
                                value={config.training.learning_rate}
                                onChange={(e) => setConfig({ ...config, training: { ...config.training, learning_rate: parseFloat(e.target.value) } })}
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Settings;
