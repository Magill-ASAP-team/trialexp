import React, { useEffect, useRef, useState } from 'react';
import Plot from 'react-plotly.js';

type Data = {
    x: number[];
    y: number[];
};

const Dashboard: React.FC = () => {
    const [data, setData] = useState<Data>({ x: [], y: [] });
    const socketRef = useRef<WebSocket | null>(null);

    useEffect(() => {
        // Initialize WebSocket connection
        socketRef.current = new WebSocket('ws://localhost:8000/ws');

        if (socketRef.current) {
            socketRef.current.onopen = () => {
                console.log('WebSocket connection opened');
            };

            socketRef.current.onmessage = (event: MessageEvent) => {
                const updatedData: Data = JSON.parse(event.data);
                setData(updatedData);
            };

            socketRef.current.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            socketRef.current.onclose = () => {
                console.log('WebSocket connection closed');
            };
        }

        return () => {
            // Clean up WebSocket connection
            if (socketRef.current) {
                socketRef.current.close();
            }
        };
    }, []);

    const handleRelayout = (event: any) => {
        if (event["xaxis.range[0]"] && event["xaxis.range[1]"]) {
            const zoomRange = [
                Math.floor(event["xaxis.range[0]"] as number),
                Math.ceil(event["xaxis.range[1]"] as number)
            ];
            if (socketRef.current) {
                socketRef.current.send(JSON.stringify({ zoom: zoomRange }));
            }
        }
    };

    return (
        <div>
            <h1>Signal Dashboard</h1>
            <Plot
                data={[{
                    x: data.x,
                    y: data.y,
                    type: 'scatter',
                    mode: 'lines+markers',
                    marker: { color: 'blue' },
                }]}
                layout={{ title: 'Zoomable Signal Plot' }}
                onRelayout={handleRelayout}
            />
        </div>
    );
};

export default Dashboard;
