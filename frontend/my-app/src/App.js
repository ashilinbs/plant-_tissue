import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
    const [formData, setFormData] = useState({
        Explant_Type: '',
        Plant_Species: '',
        Medium_Composition: '',
        Temperature: '',
        Humidity: '',
        Light_Intensity: '',
        Culture_Duration: ''
    });

    const [prediction, setPrediction] = useState(null);
    const [predictedPercentageRange, setPredictedPercentageRange] = useState('');

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const response = await axios.post('http://127.0.0.1:5000/predict', formData);
            setPrediction(response.data.predicted_class);
            setPredictedPercentageRange(response.data.predicted_percentage_range);
        } catch (error) {
            console.error("Error making prediction request", error);
        }
    };

    return (
        <div>
            <h2>Plant Tissue Culture Prediction</h2>
            <form onSubmit={handleSubmit}>
                <label>Explant Type:</label>
                <input type="text" name="Explant_Type" value={formData.Explant_Type} onChange={handleChange} />
                
                <label>Plant Species:</label>
                <input type="text" name="Plant_Species" value={formData.Plant_Species} onChange={handleChange} />
                
                <label>Medium Composition:</label>
                <input type="text" name="Medium_Composition" value={formData.Medium_Composition} onChange={handleChange} />
                
                <label>Temperature:</label>
                <input type="number" name="Temperature" value={formData.Temperature} onChange={handleChange} />
                
                <label>Humidity:</label>
                <input type="number" name="Humidity" value={formData.Humidity} onChange={handleChange} />
                
                <label>Light Intensity:</label>
                <input type="number" name="Light_Intensity" value={formData.Light_Intensity} onChange={handleChange} />
                
                <label>Culture Duration:</label>
                <input type="number" name="Culture_Duration" value={formData.Culture_Duration} onChange={handleChange} />
                
                <button type="submit">Predict</button>
            </form>

            {prediction !== null && (
                <div>
                    <h3>Predicted Regeneration Success Rate Class: {prediction}</h3>
                    <h4>Estimated Success Rate: {predictedPercentageRange}</h4>
                </div>
            )}
        </div>
    );
}

export default App;
