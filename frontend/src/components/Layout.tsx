import React, { useRef } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { LayoutDashboard, Activity, Play, Settings, Terminal } from 'lucide-react';
import { useGSAP } from '@gsap/react';
import gsap from 'gsap';

gsap.registerPlugin(useGSAP);

const Layout = () => {
    const containerRef = useRef<HTMLDivElement>(null);

    useGSAP(() => {
        gsap.from('.nav-item', {
            x: -20,
            opacity: 0,
            stagger: 0.1,
            duration: 0.5,
            ease: 'power2.out'
        });
    }, { scope: containerRef });

    const navItems = [
        { to: '/dashboard', icon: LayoutDashboard, label: 'Dashboard' },
        { to: '/training', icon: Activity, label: 'Training' },
        { to: '/inference', icon: Play, label: 'Inference' },
        { to: '/settings', icon: Settings, label: 'Settings' },
    ];

    return (
        <div className="flex min-h-screen" ref={containerRef}>
            {/* Sidebar */}
            <aside className="w-64 glass border-r border-white/10 fixed h-full z-10 flex flex-col">
                <div className="p-6 border-b border-white/10">
                    <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                        CopyAir
                    </h1>
                    <p className="text-xs text-gray-400 mt-1">AI Training Studio</p>
                </div>

                <nav className="flex-1 p-4 space-y-2">
                    {navItems.map((item) => (
                        <NavLink
                            key={item.to}
                            to={item.to}
                            className={({ isActive }) =>
                                `nav-item flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200 ${isActive
                                    ? 'bg-white/10 text-white shadow-lg backdrop-blur-sm'
                                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                                }`
                            }
                        >
                            <item.icon size={20} />
                            <span className="font-medium">{item.label}</span>
                        </NavLink>
                    ))}
                </nav>

                <div className="p-4 border-t border-white/10">
                    <div className="flex items-center gap-2 text-xs text-gray-500">
                        <Terminal size={12} />
                        <span>v1.0.0 • Local</span>
                    </div>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 ml-64 p-8">
                <div className="max-w-6xl mx-auto animate-fade-in">
                    <Outlet />
                </div>
            </main>
        </div>
    );
};

export default Layout;
