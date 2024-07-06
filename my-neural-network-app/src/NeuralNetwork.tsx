import React, { useState, useEffect } from 'react';

interface WeightsType {
  w1: number[][];
  w2: number[];
}

interface BiasesType {
  b1: number[];
  b2: number[];
}

{/*1. Couleur des lignes :
Les lignes vertes (#34a853) représentent des poids positifs. Cela signifie que la connexion entre les neurones a un effet excitateur.
Les lignes rouges (#ea4335) représentent des poids négatifs. Cela signifie que la connexion entre les neurones a un effet inhibiteur.

2. Épaisseur des lignes :
L'épaisseur des lignes est proportionnelle à la valeur absolue du poids. Plus le poids est grand (en valeur absolue), plus la ligne est épaisse. Cela permet de visualiser l'importance relative des connexions :
Une ligne épaisse indique un poids élevé (positif ou négatif), ce qui signifie que la connexion a un impact significatif sur l'activation du neurone suivant.
Une ligne fine indique un poids faible, ce qui signifie que la connexion a un impact moindre.
*/}

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
    <circle cx={x} cy={y} r={radius} style={{ fill: '#4a90e2' }} />
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
    <svg width="100%" height="100vh" viewBox="0 0 800 2000" style={{border: '1px solid #ccc'}}>
      {/* Input Layer */}
      {weights.w1.map((_, i) => (
        <Neuron key={`input-${i}`} x={50} y={50 + i * 40} />
      ))}
      
      {/* Hidden Layer */}
      {biases.b1.map((_, i) => (
        <Neuron key={`hidden-${i}`} x={400} y={50 + i * 40} />
      ))}
      
      {/* Output Layer */}
      <Neuron x={750} y={50 + biases.b1.length * 20} />
      
      {/* Connections from Input to Hidden */}
      {weights.w1.map((neuronWeights, i) => 
        neuronWeights.map((weight, j) => (
          <Connection 
            key={`w1-${i}-${j}`}
            startX={70} 
            startY={50 + i * 40} 
            endX={380} 
            endY={50 + j * 40} 
            weight={weight} 
          />
        ))
      )}
      
      {/* Connections from Hidden to Output */}
      {weights.w2.map((weight, i) => (
        <Connection 
          key={`w2-${i}`}
          startX={420} 
          startY={50 + i * 40} 
          endX={730} 
          endY={50 + biases.b1.length * 20} 
          weight={weight} 
        />
      ))}
      
      {/* Layer Labels */}
      <text x="50" y="30" textAnchor="middle">Input</text>
      <text x="400" y="30" textAnchor="middle">Hidden</text>
      <text x="750" y={70 + biases.b1.length * 20} textAnchor="middle">Output</text>
    </svg>
  );
};

export default NeuralNetwork;