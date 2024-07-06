import React, { useState, useEffect } from 'react';

interface WeightsType {
  w1: number[][];
  w2: number[];
}

interface BiasesType {
  b1: number[];
  b2: number[];
}

const NeuralNetwork: React.FC = () => {
  const [weights, setWeights] = useState<WeightsType | null>(null);
  const [biases, setBiases] = useState<BiasesType | null>(null);

  useEffect(() => {
    fetch('/model_params.json')
      .then(response => response.json())
      .then(data => {
        setWeights({ w1: data.W1, w2: data.W2 });
        setBiases({ b1: data.b1, b2: data.b2 });
      })
      .catch(error => console.error('Error loading model parameters:', error));
  }, []);

  const Neuron: React.FC<{ x: number; y: number; radius?: number }> = ({ x, y, radius = 20 }) => (
    <circle cx={x} cy={y} r={radius} fill="#4a90e2" />
  );

  const Connection: React.FC<{ startX: number; startY: number; endX: number; endY: number; weight: number }> = 
    ({ startX, startY, endX, endY, weight }) => {
    const color = weight > 0 ? "#34a853" : "#ea4335";
    const strokeWidth = Math.abs(weight) * 3;
    return (
      <line 
        x1={startX} 
        y1={startY} 
        x2={endX} 
        y2={endY} 
        stroke={color} 
        strokeWidth={strokeWidth} 
      />
    );
  };

  if (!weights || !biases) {
    return <div>Loading...</div>;
  }

  return (
    <svg width="800" height="600" style={{border: '1px solid #ccc'}}>
      {/* Input Layer */}
      {weights.w1.map((_, i) => (
        <Neuron key={`input-${i}`} x={50} y={50 + i * 40} />
      ))}
      
      {/* Hidden Layer */}
      {biases.b1.map((_, i) => (
        <Neuron key={`hidden-${i}`} x={400} y={20 + i * 18} />
      ))}
      
      {/* Output Layer */}
      <Neuron x={750} y={300} />
      
      {/* Connections from Input to Hidden */}
      {weights.w1.map((neuronWeights, i) => 
        neuronWeights.map((weight, j) => (
          <Connection 
            key={`w1-${i}-${j}`}
            startX={70} 
            startY={50 + i * 40} 
            endX={380} 
            endY={20 + j * 18} 
            weight={weight} 
          />
        ))
      )}
      
      {/* Connections from Hidden to Output */}
      {weights.w2.map((weight, i) => (
        <Connection 
          key={`w2-${i}`}
          startX={420} 
          startY={20 + i * 18} 
          endX={730} 
          endY={300} 
          weight={weight} 
        />
      ))}
      
      {/* Layer Labels */}
      <text x="50" y="30" textAnchor="middle">Input</text>
      <text x="400" y="10" textAnchor="middle">Hidden</text>
      <text x="750" y="330" textAnchor="middle">Output</text>
    </svg>
  );
};

export default NeuralNetwork;