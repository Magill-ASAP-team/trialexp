import { useState, useEffect } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import Dashboard from './Dashboard'

function App() {
  const [count, setCount] = useState(0)
  const [dropdownOptions, setDropdownOptions] = useState({
    dropdown1: [],
    dropdown2: [],
    dropdown3: []
  });
  const [selectedDropdown1, setSelectedDropdown1] = useState('');

  useEffect(() => {
    fetch('http://localhost:8000/dropdown-options')
      .then(response => response.json())
      .then(data => setDropdownOptions(data));
  }, []);

  useEffect(() => {
    if (selectedDropdown1) {
      // Fetch new options for Dropdown 2 based on the selection of Dropdown 1
      fetch(`http://localhost:8000/dropdown-options?dropdown1=${selectedDropdown1}`)
        .then(response => response.json())
        .then(data => setDropdownOptions(prevOptions => ({
          ...prevOptions,
          dropdown2: data.dropdown2 || []
        })));
    }
  }, [selectedDropdown1]);

  return (
    <div style={{ display: 'flex' }}>
      <div style={{ width: '20%', padding: '10px', borderRight: '1px solid #ccc' }}>
        <h2>Sidebar</h2>
        <div>
          <label htmlFor="dropdown1">Dropdown 1:</label>
          <select id="dropdown1" onChange={(e) => setSelectedDropdown1(e.target.value)}>
            {dropdownOptions.dropdown1.map((option, index) => (
              <option key={index} value={option}>{option}</option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="dropdown2">Dropdown 2:</label>
          <select id="dropdown2">
            {Array.isArray(dropdownOptions.dropdown2) && dropdownOptions.dropdown2.map((option, index) => (
              <option key={index} value={option}>{option}</option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor="dropdown3">Dropdown 3:</label>
          <select id="dropdown3">
            {dropdownOptions.dropdown3.map((option, index) => (
              <option key={index} value={option}>{option}</option>
            ))}
          </select>
        </div>
      </div>
      <div style={{ width: '80%', padding: '10px' }}>
        <Dashboard/>
      </div>
    </div>
  )
}

export default App
