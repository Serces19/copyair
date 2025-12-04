import React, { useEffect, useState } from 'react';
import { getConfig, updateConfig } from '../api/client';
import {
    Save,
    Cpu,
    Activity,
    Layers,
    Image as ImageIcon,
    Eye,
    Zap,
    ChevronDown,
    ChevronUp,
    Database
} from 'lucide-react';

// Helper component for collapsible sections
const Section = ({ title, icon: Icon, children, defaultOpen = false }: any) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);
    return (
        <div className="card overflow-hidden transition-all duration-300">
            <button
                className="w-full flex items-center justify-between p-2 hover:bg-white/5 rounded-lg transition-colors"
                onClick={() => setIsOpen(!isOpen)}
            >
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-blue-500/10 rounded-lg text-blue-400">
                        <Icon size={20} />
                    </div>
                    <h3 className="text-lg font-semibold">{title}</h3>
                </div>
                {isOpen ? <ChevronUp size={20} className="text-gray-500" /> : <ChevronDown size={20} className="text-gray-500" />}
            </button>

            <div className={`grid transition-all duration-300 ${isOpen ? 'grid-rows-[1fr] opacity-100 mt-4' : 'grid-rows-[0fr] opacity-0'}`}>
                <div className="overflow-hidden">
                    {children}
                </div>
            </div>
        </div>
    );
};

// Helper for input fields
const InputField = ({ label, value, onChange, type = "text", options = null, step = "any" }: any) => (
    <div className="flex flex-col gap-1.5">
        <label className="text-xs font-medium text-gray-400 uppercase tracking-wider">{label}</label>
        {options ? (
            <select
                className="input bg-black/20 border-white/10 focus:border-blue-500 transition-colors"
                value={value}
                onChange={(e) => onChange(e.target.value)}
            >
                {options.map((opt: string) => (
                    <option key={opt} value={opt}>{opt}</option>
                ))}
            </select>
        ) : type === 'checkbox' ? (
            <div className="flex items-center h-10">
                <label className="relative inline-flex items-center cursor-pointer">
                    <input
                        type="checkbox"
                        className="sr-only peer"
                        checked={value}
                        onChange={(e) => onChange(e.target.checked)}
                    />
                    <div className="w-11 h-6 bg-gray-700 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-500 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                    <span className="ml-3 text-sm font-medium text-gray-300">{value ? 'Enabled' : 'Disabled'}</span>
                </label>
            </div>
        ) : (
            <input
                type={type}
                step={step}
                className="input bg-black/20 border-white/10 focus:border-blue-500 transition-colors"
                value={value}
                onChange={(e) => onChange(type === 'number' ? parseFloat(e.target.value) : e.target.value)}
            />
        )}
    </div>
);

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

    // Helper to update nested state
    const update = (path: string, value: any) => {
        const keys = path.split('.');
        setConfig((prev: any) => {
            const newConfig = { ...prev };
            let current = newConfig;
            for (let i = 0; i < keys.length - 1; i++) {
                current = current[keys[i]];
            }
            current[keys[keys.length - 1]] = value;
            return newConfig;
        });
    };

    if (loading || !config) return (
        <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        </div>
    );

    return (
        <div className="pb-20">
            <div className="flex justify-between items-center mb-8 sticky top-0 bg-black/80 backdrop-blur-md py-4 z-20 border-b border-white/5">
                <div>
                    <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">Settings</h2>
                    <p className="text-gray-400 text-sm mt-1">Manage global parameters for training and inference</p>
                </div>
                <button onClick={handleSave} className="btn btn-primary shadow-lg shadow-blue-500/20">
                    <Save size={18} />
                    Save Changes
                </button>
            </div>

            <div className="space-y-6 max-w-4xl mx-auto">

                {/* 1. Model Architecture */}
                <Section title="Model Architecture" icon={Cpu} defaultOpen={true}>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <InputField
                            label="Architecture"
                            value={config.model.architecture}
                            onChange={(v: any) => update('model.architecture', v)}
                            options={['unet', 'nafnet', 'convnext', 'mambair']}
                        />
                        <InputField
                            label="Model Size"
                            value={config.model.size}
                            onChange={(v: any) => update('model.size', v)}
                            options={['nano', 'tiny', 'base', 'small', 'large']}
                        />
                        <InputField
                            label="Activation Function"
                            value={config.model.activation}
                            onChange={(v: any) => update('model.activation', v)}
                            options={['relu', 'gelu', 'mish', 'silu', 'leaky_relu']}
                        />
                        <InputField
                            label="Base Channels"
                            value={config.model.base_channels}
                            onChange={(v: any) => update('model.base_channels', v)}
                            type="number"
                        />
                        <InputField
                            label="Use Batch Norm"
                            value={config.model.use_batchnorm}
                            onChange={(v: any) => update('model.use_batchnorm', v)}
                            type="checkbox"
                        />
                        <InputField
                            label="Use Transpose Conv"
                            value={config.model.use_transpose}
                            onChange={(v: any) => update('model.use_transpose', v)}
                            type="checkbox"
                        />
                    </div>
                </Section>

                {/* 2. Training Parameters */}
                <Section title="Training & Optimization" icon={Activity}>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <InputField
                            label="Epochs"
                            value={config.training.epochs}
                            onChange={(v: any) => update('training.epochs', v)}
                            type="number"
                        />
                        <InputField
                            label="Batch Size"
                            value={config.training.batch_size}
                            onChange={(v: any) => update('training.batch_size', v)}
                            type="number"
                        />
                        <InputField
                            label="Learning Rate"
                            value={config.training.learning_rate}
                            onChange={(v: any) => update('training.learning_rate', v)}
                            type="number"
                            step="0.00001"
                        />
                        <InputField
                            label="Optimizer"
                            value={config.training.optimizer.type}
                            onChange={(v: any) => update('training.optimizer.type', v)}
                            options={['adam', 'adamw', 'sgd', 'rmsprop']}
                        />
                        <InputField
                            label="Scheduler"
                            value={config.training.scheduler.type}
                            onChange={(v: any) => update('training.scheduler.type', v)}
                            options={['cosine', 'step', 'plateau']}
                        />
                        <InputField
                            label="Mixed Precision"
                            value={config.training.mixed_precision}
                            onChange={(v: any) => update('training.mixed_precision', v)}
                            type="checkbox"
                        />
                    </div>
                </Section>

                {/* 3. Loss Functions */}
                <Section title="Loss Functions" icon={Layers}>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <InputField
                            label="L1 Loss (Lambda)"
                            value={config.loss.lambda_l1}
                            onChange={(v: any) => update('loss.lambda_l1', v)}
                            type="number" step="0.1"
                        />
                        <InputField
                            label="SSIM Loss (Lambda)"
                            value={config.loss.lambda_ssim}
                            onChange={(v: any) => update('loss.lambda_ssim', v)}
                            type="number" step="0.1"
                        />
                        <InputField
                            label="Perceptual Loss (Lambda)"
                            value={config.loss.lambda_perceptual}
                            onChange={(v: any) => update('loss.lambda_perceptual', v)}
                            type="number" step="0.1"
                        />
                        <InputField
                            label="Multiscale Loss"
                            value={config.loss.lambda_multiscale}
                            onChange={(v: any) => update('loss.lambda_multiscale', v)}
                            type="number" step="0.1"
                        />
                        <InputField
                            label="Sobel Loss"
                            value={config.loss.lambda_sobel}
                            onChange={(v: any) => update('loss.lambda_sobel', v)}
                            type="number" step="0.1"
                        />
                        <InputField
                            label="Charbonnier Loss"
                            value={config.loss.lambda_charbonnier}
                            onChange={(v: any) => update('loss.lambda_charbonnier', v)}
                            type="number" step="0.1"
                        />
                    </div>
                </Section>

                {/* 4. Data Augmentation */}
                <Section title="Data Augmentation" icon={ImageIcon}>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="md:col-span-3">
                            <InputField
                                label="Enable Augmentation"
                                value={config.augmentation.enabled}
                                onChange={(v: any) => update('augmentation.enabled', v)}
                                type="checkbox"
                            />
                        </div>
                        <InputField
                            label="Image Size"
                            value={config.augmentation.img_size}
                            onChange={(v: any) => update('augmentation.img_size', v)}
                            type="number"
                        />
                        <InputField
                            label="Horizontal Flip Prob"
                            value={config.augmentation.horizontal_flip_p}
                            onChange={(v: any) => update('augmentation.horizontal_flip_p', v)}
                            type="number" step="0.1"
                        />
                        <InputField
                            label="Vertical Flip Prob"
                            value={config.augmentation.vertical_flip_p}
                            onChange={(v: any) => update('augmentation.vertical_flip_p', v)}
                            type="number" step="0.1"
                        />
                        <InputField
                            label="Color Jitter Prob"
                            value={config.augmentation.color_jitter_p}
                            onChange={(v: any) => update('augmentation.color_jitter_p', v)}
                            type="number" step="0.1"
                        />
                        <InputField
                            label="Gaussian Blur Prob"
                            value={config.augmentation.gaussian_blur_p}
                            onChange={(v: any) => update('augmentation.gaussian_blur_p', v)}
                            type="number" step="0.1"
                        />
                    </div>
                </Section>

                {/* 5. Masked Loss */}
                <Section title="Masked Loss (Advanced)" icon={Eye}>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <InputField
                            label="Enable Masked Loss"
                            value={config.masked_loss.enabled}
                            onChange={(v: any) => update('masked_loss.enabled', v)}
                            type="checkbox"
                        />
                        <InputField
                            label="Threshold"
                            value={config.masked_loss.threshold}
                            onChange={(v: any) => update('masked_loss.threshold', v)}
                            type="number" step="0.01"
                        />
                    </div>
                </Section>

                {/* 6. System & Logging */}
                <Section title="System & Logging" icon={Database}>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <InputField
                            label="Device"
                            value={config.device}
                            onChange={(v: any) => update('device', v)}
                            options={['cuda', 'cpu']}
                        />
                        <InputField
                            label="Num Workers"
                            value={config.num_workers}
                            onChange={(v: any) => update('num_workers', v)}
                            type="number"
                        />
                        <InputField
                            label="MLflow Enabled"
                            value={config.mlflow.enabled}
                            onChange={(v: any) => update('mlflow.enabled', v)}
                            type="checkbox"
                        />
                        <InputField
                            label="Experiment Name"
                            value={config.mlflow.experiment_name}
                            onChange={(v: any) => update('mlflow.experiment_name', v)}
                        />
                    </div>
                </Section>

            </div>
        </div>
    );
};

export default Settings;
