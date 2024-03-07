import { useState } from 'react';
import './styles.css';

function App() {
  const [deviceId, setDeviceId] = useState('none');
  const [dataFormat, setDataFormat] = useState('data');
  const [apiResult, setApiResult] = useState(null); 

  const handleSubmit = async (event) => {
    event.preventDefault();

    try {
      const response = await fetch('backend url', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ deviceId, dataFormat })
      });
      
      if (response.ok) {
        const result = await response.json(); 
        setApiResult(result); 
      } else {
        console.error('Failed to fetch data from the API');
      }
    } catch (error) {
      console.error('Error fetching data from the API:', error);
    }
  };

  const handleDataFormatChange = (event) => {
    setDataFormat(event.target.value);
  };

  return (
    <>
      <form className="totalForm" onSubmit={handleSubmit}>
        <div className="form-box">
          <label htmlFor="device">Device Id</label>
          <input
            type="text"
            id="device"
            value={deviceId}
            onChange={(e) => setDeviceId(e.target.value)}
          />
        </div>

        <div className="form-select">
          <label htmlFor="dataFormat">Select a Dataset:</label>
          <select
            id="dataFormat"
            value={dataFormat}
            onChange={handleDataFormatChange}
          >
            <option value="data1">Consignment Dataset</option>
            <option value="data2">Item Count Dataset</option>
          </select>
        </div>
        
        <button type="submit" className="btn">
          Submit
        </button>
      </form>

      
      {apiResult && (
        <div className="api-result">
          <h2>Results:</h2>
          <img src={`data:image/png;base64,${apiResult.currentGraph}`} alt="Current Graph"></img>
          <img src={`data:image/png;base64,${apiResult.predictionGraph}`} alt="Prediction Graph"></img>
          <p>Error: {apiResult.error}</p>
          <p>If error rate is samaller then 1 it is a good prediction</p>
        </div>
      )}
    </>
  );
}

export default App;
