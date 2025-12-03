import React from 'react';

const Dashboard = () => {
    return (
        <div>
            <h2 className="text-3xl font-bold mb-6">Dashboard</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="card">
                    <h3 className="text-gray-400 text-sm font-medium mb-2">Model Status</h3>
                    <p className="text-2xl font-bold text-green-400">Ready</p>
                </div>
                <div className="card">
                    <h3 className="text-gray-400 text-sm font-medium mb-2">Last Training</h3>
                    <p className="text-2xl font-bold">2h ago</p>
                </div>
                <div className="card">
                    <h3 className="text-gray-400 text-sm font-medium mb-2">GPU Usage</h3>
                    <p className="text-2xl font-bold text-blue-400">Idle</p>
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
